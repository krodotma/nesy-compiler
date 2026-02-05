#!/usr/bin/env python3
"""
documentation.py - Documentation System (Step 44)

Comprehensive documentation system for Research Agent including
API documentation, guides, and auto-generated reference docs.

PBTSO Phase: DOCUMENT

Bus Topics:
- a2a.research.docs.generate
- a2a.research.docs.update
- research.docs.build

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import ast
import fcntl
import inspect
import json
import os
import re
import socket
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class DocFormat(Enum):
    """Documentation output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    RST = "rst"


class DocType(Enum):
    """Types of documentation."""
    API = "api"
    GUIDE = "guide"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    CHANGELOG = "changelog"


@dataclass
class DocConfig:
    """Configuration for documentation system."""

    output_dir: str = ""
    format: DocFormat = DocFormat.MARKDOWN
    include_private: bool = False
    include_source: bool = True
    include_examples: bool = True
    max_depth: int = 3
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if not self.output_dir:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.output_dir = f"{pluribus_root}/.pluribus/research/docs"
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Parameter:
    """Documentation for a function/method parameter."""

    name: str
    type: str = "Any"
    description: str = ""
    default: Optional[str] = None
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "default": self.default,
            "required": self.required,
        }


@dataclass
class ReturnValue:
    """Documentation for a return value."""

    type: str
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "description": self.description,
        }


@dataclass
class Example:
    """A code example."""

    code: str
    description: str = ""
    language: str = "python"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "description": self.description,
            "language": self.language,
        }


@dataclass
class FunctionDoc:
    """Documentation for a function or method."""

    name: str
    signature: str
    description: str = ""
    parameters: List[Parameter] = field(default_factory=list)
    returns: Optional[ReturnValue] = None
    raises: List[str] = field(default_factory=list)
    examples: List[Example] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    source_file: str = ""
    line_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "signature": self.signature,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": self.returns.to_dict() if self.returns else None,
            "raises": self.raises,
            "examples": [e.to_dict() for e in self.examples],
            "is_async": self.is_async,
            "source_file": self.source_file,
            "line_number": self.line_number,
        }


@dataclass
class ClassDoc:
    """Documentation for a class."""

    name: str
    description: str = ""
    bases: List[str] = field(default_factory=list)
    attributes: List[Parameter] = field(default_factory=list)
    methods: List[FunctionDoc] = field(default_factory=list)
    class_methods: List[FunctionDoc] = field(default_factory=list)
    static_methods: List[FunctionDoc] = field(default_factory=list)
    properties: List[Parameter] = field(default_factory=list)
    examples: List[Example] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "bases": self.bases,
            "attributes": [a.to_dict() for a in self.attributes],
            "methods": [m.to_dict() for m in self.methods],
            "class_methods": [m.to_dict() for m in self.class_methods],
            "static_methods": [m.to_dict() for m in self.static_methods],
            "properties": [p.to_dict() for p in self.properties],
            "examples": [e.to_dict() for e in self.examples],
        }


@dataclass
class ModuleDoc:
    """Documentation for a module."""

    name: str
    path: str
    description: str = ""
    functions: List[FunctionDoc] = field(default_factory=list)
    classes: List[ClassDoc] = field(default_factory=list)
    constants: List[Parameter] = field(default_factory=list)
    submodules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "constants": [c.to_dict() for c in self.constants],
            "submodules": self.submodules,
        }


@dataclass
class APIEndpoint:
    """Documentation for an API endpoint."""

    path: str
    method: str
    description: str = ""
    parameters: List[Parameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    examples: List[Example] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "request_body": self.request_body,
            "responses": self.responses,
            "examples": [e.to_dict() for e in self.examples],
            "tags": self.tags,
            "deprecated": self.deprecated,
        }


# ============================================================================
# Documentation Extractors
# ============================================================================


class DocstringParser:
    """Parser for Python docstrings."""

    @staticmethod
    def parse(docstring: Optional[str]) -> Dict[str, Any]:
        """Parse a docstring into structured data."""
        if not docstring:
            return {"description": "", "params": [], "returns": None, "raises": [], "examples": []}

        lines = docstring.strip().split("\n")
        result = {
            "description": "",
            "params": [],
            "returns": None,
            "raises": [],
            "examples": [],
        }

        current_section = "description"
        current_content: List[str] = []
        current_param_name = ""

        for line in lines:
            stripped = line.strip()

            # Section headers
            if stripped in ("Args:", "Arguments:", "Parameters:"):
                result["description"] = "\n".join(current_content).strip()
                current_section = "params"
                current_content = []
                continue
            elif stripped in ("Returns:", "Return:"):
                current_section = "returns"
                current_content = []
                continue
            elif stripped in ("Raises:", "Raise:", "Exceptions:"):
                current_section = "raises"
                current_content = []
                continue
            elif stripped in ("Example:", "Examples:"):
                current_section = "examples"
                current_content = []
                continue
            elif stripped in ("Note:", "Notes:"):
                current_section = "notes"
                current_content = []
                continue

            # Process content based on section
            if current_section == "description":
                current_content.append(line)
            elif current_section == "params":
                # Parse parameter: "name (type): description" or "name: description"
                param_match = re.match(r"(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.+)", stripped)
                if param_match:
                    name, param_type, desc = param_match.groups()
                    result["params"].append({
                        "name": name,
                        "type": param_type or "Any",
                        "description": desc,
                    })
                    current_param_name = name
                elif current_param_name and stripped:
                    # Continuation of previous param
                    if result["params"]:
                        result["params"][-1]["description"] += " " + stripped
            elif current_section == "returns":
                if stripped:
                    if result["returns"] is None:
                        # Parse return: "type: description" or just "description"
                        ret_match = re.match(r"([^:]+):\s*(.+)", stripped)
                        if ret_match:
                            result["returns"] = {
                                "type": ret_match.group(1).strip(),
                                "description": ret_match.group(2).strip(),
                            }
                        else:
                            result["returns"] = {
                                "type": "Any",
                                "description": stripped,
                            }
                    else:
                        result["returns"]["description"] += " " + stripped
            elif current_section == "raises":
                exc_match = re.match(r"(\w+):\s*(.+)", stripped)
                if exc_match:
                    result["raises"].append(f"{exc_match.group(1)}: {exc_match.group(2)}")
                elif stripped:
                    result["raises"].append(stripped)
            elif current_section == "examples":
                current_content.append(line)

        # Final processing
        if current_section == "description":
            result["description"] = "\n".join(current_content).strip()
        elif current_section == "examples":
            example_code = "\n".join(current_content).strip()
            if example_code:
                result["examples"].append({"code": example_code})

        return result


class ModuleExtractor:
    """Extracts documentation from Python modules."""

    def __init__(self, config: DocConfig):
        self.config = config
        self._parser = DocstringParser()

    def extract_module(self, module_path: str) -> ModuleDoc:
        """Extract documentation from a module file."""
        path = Path(module_path)
        if not path.exists():
            return ModuleDoc(name=path.stem, path=module_path)

        source = path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ModuleDoc(name=path.stem, path=module_path)

        doc = ModuleDoc(
            name=path.stem,
            path=module_path,
        )

        # Get module docstring
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            doc.description = tree.body[0].value.value.strip()

        # Extract module-level items
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Skip nested functions
                if hasattr(node, "col_offset") and node.col_offset == 0:
                    func_doc = self._extract_function(node, source)
                    if func_doc:
                        doc.functions.append(func_doc)

            elif isinstance(node, ast.ClassDef):
                # Skip nested classes
                if hasattr(node, "col_offset") and node.col_offset == 0:
                    class_doc = self._extract_class(node, source)
                    if class_doc:
                        doc.classes.append(class_doc)

            elif isinstance(node, ast.Assign):
                # Extract module-level constants
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name.isupper() or not name.startswith("_"):
                            doc.constants.append(Parameter(
                                name=name,
                                type=self._infer_type(node.value),
                            ))

        return doc

    def _extract_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        source: str,
    ) -> Optional[FunctionDoc]:
        """Extract documentation from a function."""
        if node.name.startswith("_") and not self.config.include_private:
            return None

        docstring = ast.get_docstring(node) or ""
        parsed = self._parser.parse(docstring)

        # Build signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        signature = f"{node.name}({', '.join(args)})"
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"

        # Build parameters
        parameters = []
        defaults_offset = len(node.args.args) - len(node.args.defaults)

        for i, arg in enumerate(node.args.args):
            if arg.arg == "self" or arg.arg == "cls":
                continue

            param = Parameter(name=arg.arg)

            if arg.annotation:
                param.type = ast.unparse(arg.annotation)

            # Check for default value
            default_idx = i - defaults_offset
            if default_idx >= 0 and default_idx < len(node.args.defaults):
                param.default = ast.unparse(node.args.defaults[default_idx])
                param.required = False

            # Add description from docstring
            for doc_param in parsed.get("params", []):
                if doc_param["name"] == arg.arg:
                    param.description = doc_param["description"]
                    if doc_param.get("type"):
                        param.type = doc_param["type"]
                    break

            parameters.append(param)

        # Build return value
        returns = None
        if parsed.get("returns"):
            returns = ReturnValue(
                type=parsed["returns"].get("type", "Any"),
                description=parsed["returns"].get("description", ""),
            )
        elif node.returns:
            returns = ReturnValue(type=ast.unparse(node.returns))

        # Build examples
        examples = [
            Example(code=ex.get("code", ""), description=ex.get("description", ""))
            for ex in parsed.get("examples", [])
        ]

        return FunctionDoc(
            name=node.name,
            signature=signature,
            description=parsed.get("description", ""),
            parameters=parameters,
            returns=returns,
            raises=parsed.get("raises", []),
            examples=examples,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            line_number=node.lineno,
        )

    def _extract_class(self, node: ast.ClassDef, source: str) -> Optional[ClassDoc]:
        """Extract documentation from a class."""
        if node.name.startswith("_") and not self.config.include_private:
            return None

        docstring = ast.get_docstring(node) or ""
        parsed = self._parser.parse(docstring)

        # Get base classes
        bases = []
        for base in node.bases:
            bases.append(ast.unparse(base))

        class_doc = ClassDoc(
            name=node.name,
            description=parsed.get("description", ""),
            bases=bases,
            line_number=node.lineno,
        )

        # Extract examples from docstring
        class_doc.examples = [
            Example(code=ex.get("code", ""), description=ex.get("description", ""))
            for ex in parsed.get("examples", [])
        ]

        # Extract methods and attributes
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_doc = self._extract_function(item, source)
                if func_doc:
                    if item.name == "__init__":
                        # Extract attributes from __init__
                        pass
                    elif any(
                        isinstance(d, ast.Name) and d.id == "classmethod"
                        for d in item.decorator_list
                    ):
                        class_doc.class_methods.append(func_doc)
                    elif any(
                        isinstance(d, ast.Name) and d.id == "staticmethod"
                        for d in item.decorator_list
                    ):
                        class_doc.static_methods.append(func_doc)
                    elif any(
                        isinstance(d, ast.Name) and d.id == "property"
                        for d in item.decorator_list
                    ):
                        class_doc.properties.append(Parameter(
                            name=func_doc.name,
                            type=func_doc.returns.type if func_doc.returns else "Any",
                            description=func_doc.description,
                        ))
                    else:
                        class_doc.methods.append(func_doc)

            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                # Class attribute with annotation
                class_doc.attributes.append(Parameter(
                    name=item.target.id,
                    type=ast.unparse(item.annotation) if item.annotation else "Any",
                ))

        return class_doc

    def _infer_type(self, node: ast.expr) -> str:
        """Infer type from AST node."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "List"
        elif isinstance(node, ast.Dict):
            return "Dict"
        elif isinstance(node, ast.Set):
            return "Set"
        elif isinstance(node, ast.Tuple):
            return "Tuple"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
        return "Any"


# ============================================================================
# Documentation Generators
# ============================================================================


class DocGenerator(ABC):
    """Abstract base for documentation generators."""

    @abstractmethod
    def generate(self, doc: Union[ModuleDoc, ClassDoc, FunctionDoc]) -> str:
        """Generate documentation string."""
        pass


class MarkdownGenerator(DocGenerator):
    """Generates Markdown documentation."""

    def generate(self, doc: Union[ModuleDoc, ClassDoc, FunctionDoc]) -> str:
        """Generate Markdown documentation."""
        if isinstance(doc, ModuleDoc):
            return self._generate_module(doc)
        elif isinstance(doc, ClassDoc):
            return self._generate_class(doc)
        elif isinstance(doc, FunctionDoc):
            return self._generate_function(doc)
        return ""

    def _generate_module(self, doc: ModuleDoc) -> str:
        """Generate module documentation."""
        lines = [
            f"# {doc.name}",
            "",
        ]

        if doc.description:
            lines.extend([doc.description, ""])

        # Table of contents
        if doc.classes or doc.functions:
            lines.append("## Contents\n")
            if doc.classes:
                lines.append("### Classes\n")
                for cls in doc.classes:
                    lines.append(f"- [{cls.name}](#{cls.name.lower()})")
                lines.append("")
            if doc.functions:
                lines.append("### Functions\n")
                for func in doc.functions:
                    lines.append(f"- [{func.name}](#{func.name.lower()})")
                lines.append("")

        # Classes
        for cls in doc.classes:
            lines.extend([self._generate_class(cls), ""])

        # Functions
        for func in doc.functions:
            lines.extend([self._generate_function(func), ""])

        return "\n".join(lines)

    def _generate_class(self, doc: ClassDoc) -> str:
        """Generate class documentation."""
        lines = [
            f"## {doc.name}",
            "",
        ]

        if doc.bases:
            lines.append(f"*Inherits from: {', '.join(doc.bases)}*\n")

        if doc.description:
            lines.extend([doc.description, ""])

        # Attributes
        if doc.attributes:
            lines.append("### Attributes\n")
            lines.append("| Name | Type | Description |")
            lines.append("|------|------|-------------|")
            for attr in doc.attributes:
                lines.append(f"| `{attr.name}` | `{attr.type}` | {attr.description} |")
            lines.append("")

        # Methods
        if doc.methods:
            lines.append("### Methods\n")
            for method in doc.methods:
                lines.extend([self._generate_function(method, level=4), ""])

        # Examples
        if doc.examples:
            lines.append("### Examples\n")
            for example in doc.examples:
                if example.description:
                    lines.append(example.description)
                lines.append(f"```{example.language}")
                lines.append(example.code)
                lines.append("```\n")

        return "\n".join(lines)

    def _generate_function(self, doc: FunctionDoc, level: int = 3) -> str:
        """Generate function documentation."""
        prefix = "#" * level
        lines = [
            f"{prefix} {doc.name}",
            "",
            f"```python",
            f"{'async ' if doc.is_async else ''}{doc.signature}",
            "```",
            "",
        ]

        if doc.description:
            lines.extend([doc.description, ""])

        # Parameters
        if doc.parameters:
            lines.append("**Parameters:**\n")
            for param in doc.parameters:
                req = "" if param.required else " (optional)"
                default = f" = `{param.default}`" if param.default else ""
                lines.append(f"- `{param.name}` (`{param.type}`){req}{default}: {param.description}")
            lines.append("")

        # Returns
        if doc.returns:
            lines.append("**Returns:**\n")
            lines.append(f"- `{doc.returns.type}`: {doc.returns.description}")
            lines.append("")

        # Raises
        if doc.raises:
            lines.append("**Raises:**\n")
            for exc in doc.raises:
                lines.append(f"- {exc}")
            lines.append("")

        # Examples
        if doc.examples:
            lines.append("**Examples:**\n")
            for example in doc.examples:
                if example.description:
                    lines.append(example.description)
                lines.append(f"```{example.language}")
                lines.append(example.code)
                lines.append("```\n")

        return "\n".join(lines)


class HTMLGenerator(DocGenerator):
    """Generates HTML documentation."""

    def generate(self, doc: Union[ModuleDoc, ClassDoc, FunctionDoc]) -> str:
        """Generate HTML documentation."""
        # Use markdown generator and convert
        md_gen = MarkdownGenerator()
        markdown = md_gen.generate(doc)
        return self._markdown_to_html(markdown)

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert Markdown to HTML (simplified)."""
        html = markdown

        # Headers
        html = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)

        # Code blocks
        html = re.sub(
            r"```(\w+)\n(.+?)\n```",
            r'<pre><code class="language-\1">\2</code></pre>',
            html,
            flags=re.DOTALL
        )

        # Inline code
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

        # Bold
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

        # Lists
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)

        # Paragraphs
        html = re.sub(r"\n\n", r"</p><p>", html)

        return f"<div class='documentation'><p>{html}</p></div>"


# ============================================================================
# Documentation Manager
# ============================================================================


class DocumentationManager:
    """
    Documentation manager for Research Agent.

    Features:
    - Auto-generate API documentation
    - Module/class/function extraction
    - Multiple output formats
    - Documentation building

    PBTSO Phase: DOCUMENT

    Example:
        docs = DocumentationManager()

        # Generate module documentation
        module_doc = docs.document_module("/path/to/module.py")

        # Build all documentation
        docs.build_all("/path/to/source")

        # Export
        docs.export(module_doc, "docs/module.md")
    """

    def __init__(
        self,
        config: Optional[DocConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the documentation manager.

        Args:
            config: Documentation configuration
            bus: AgentBus for event emission
        """
        self.config = config or DocConfig()
        self.bus = bus or AgentBus()

        self._extractor = ModuleExtractor(self.config)
        self._generators: Dict[DocFormat, DocGenerator] = {
            DocFormat.MARKDOWN: MarkdownGenerator(),
            DocFormat.HTML: HTMLGenerator(),
        }

        # Documentation registry
        self._modules: Dict[str, ModuleDoc] = {}
        self._api_endpoints: List[APIEndpoint] = {}

        # Statistics
        self._stats = {
            "modules_documented": 0,
            "classes_documented": 0,
            "functions_documented": 0,
            "endpoints_documented": 0,
        }

    def document_module(self, module_path: str) -> ModuleDoc:
        """
        Generate documentation for a module.

        Args:
            module_path: Path to Python module

        Returns:
            ModuleDoc with extracted documentation
        """
        doc = self._extractor.extract_module(module_path)

        self._modules[doc.name] = doc
        self._stats["modules_documented"] += 1
        self._stats["classes_documented"] += len(doc.classes)
        self._stats["functions_documented"] += len(doc.functions)

        self._emit_event("a2a.research.docs.generate", {
            "module": doc.name,
            "classes": len(doc.classes),
            "functions": len(doc.functions),
        })

        return doc

    def document_package(self, package_path: str) -> List[ModuleDoc]:
        """
        Generate documentation for a package.

        Args:
            package_path: Path to Python package

        Returns:
            List of ModuleDoc for all modules
        """
        docs: List[ModuleDoc] = []
        path = Path(package_path)

        if not path.is_dir():
            return docs

        for py_file in path.rglob("*.py"):
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue
            doc = self.document_module(str(py_file))
            docs.append(doc)

        return docs

    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """Register an API endpoint."""
        self._api_endpoints[f"{endpoint.method}:{endpoint.path}"] = endpoint
        self._stats["endpoints_documented"] += 1

    def generate(
        self,
        doc: Union[ModuleDoc, ClassDoc, FunctionDoc],
        format: Optional[DocFormat] = None,
    ) -> str:
        """
        Generate documentation in specified format.

        Args:
            doc: Documentation object
            format: Output format

        Returns:
            Generated documentation string
        """
        fmt = format or self.config.format
        generator = self._generators.get(fmt, self._generators[DocFormat.MARKDOWN])
        return generator.generate(doc)

    def export(
        self,
        doc: Union[ModuleDoc, ClassDoc, FunctionDoc],
        output_path: str,
        format: Optional[DocFormat] = None,
    ) -> None:
        """
        Export documentation to file.

        Args:
            doc: Documentation object
            output_path: Output file path
            format: Output format
        """
        content = self.generate(doc, format)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        self._emit_event("a2a.research.docs.update", {
            "path": output_path,
            "type": type(doc).__name__,
        })

    def build_all(
        self,
        source_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Build documentation for all modules in source path.

        Args:
            source_path: Path to source code
            output_path: Output directory

        Returns:
            Dict mapping module names to output paths
        """
        output_dir = Path(output_path or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, str] = {}

        # Document all modules
        docs = self.document_package(source_path)

        for doc in docs:
            # Generate output path
            relative_path = doc.path.replace(source_path, "").lstrip("/")
            output_file = output_dir / relative_path.replace(".py", ".md")

            self.export(doc, str(output_file))
            results[doc.name] = str(output_file)

        # Generate index
        index_path = output_dir / "index.md"
        self._generate_index(docs, index_path)
        results["index"] = str(index_path)

        self._emit_event("research.docs.build", {
            "source": source_path,
            "output": str(output_dir),
            "modules": len(docs),
        })

        return results

    def generate_api_docs(self, output_path: Optional[str] = None) -> str:
        """
        Generate API documentation for registered endpoints.

        Args:
            output_path: Optional output path

        Returns:
            Generated API documentation
        """
        lines = [
            "# API Documentation",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]

        # Group by tag
        by_tag: Dict[str, List[APIEndpoint]] = {}
        for endpoint in self._api_endpoints.values():
            for tag in endpoint.tags or ["default"]:
                if tag not in by_tag:
                    by_tag[tag] = []
                by_tag[tag].append(endpoint)

        for tag, endpoints in sorted(by_tag.items()):
            lines.append(f"## {tag.title()}")
            lines.append("")

            for endpoint in endpoints:
                lines.append(f"### {endpoint.method} {endpoint.path}")
                lines.append("")

                if endpoint.deprecated:
                    lines.append("> **Deprecated**")
                    lines.append("")

                if endpoint.description:
                    lines.append(endpoint.description)
                    lines.append("")

                if endpoint.parameters:
                    lines.append("**Parameters:**")
                    for param in endpoint.parameters:
                        lines.append(f"- `{param.name}` ({param.type}): {param.description}")
                    lines.append("")

                if endpoint.request_body:
                    lines.append("**Request Body:**")
                    lines.append("```json")
                    lines.append(json.dumps(endpoint.request_body, indent=2))
                    lines.append("```")
                    lines.append("")

                if endpoint.responses:
                    lines.append("**Responses:**")
                    for code, response in endpoint.responses.items():
                        lines.append(f"- `{code}`: {response.get('description', '')}")
                    lines.append("")

        content = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(content)

        return content

    def get_stats(self) -> Dict[str, Any]:
        """Get documentation statistics."""
        return {
            **self._stats,
            "registered_modules": len(self._modules),
            "registered_endpoints": len(self._api_endpoints),
        }

    def _generate_index(self, docs: List[ModuleDoc], output_path: Path) -> None:
        """Generate documentation index."""
        lines = [
            "# Research Agent Documentation",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Modules",
            "",
        ]

        for doc in sorted(docs, key=lambda d: d.name):
            relative_path = doc.name.replace(".", "/") + ".md"
            lines.append(f"- [{doc.name}]({relative_path})")
            if doc.description:
                desc = doc.description.split("\n")[0][:100]
                lines.append(f"  - {desc}")

        output_path.write_text("\n".join(lines))

    def _emit_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
    ) -> str:
        """Emit event with file locking."""
        if not self.config.emit_to_bus:
            return ""

        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "documentation",
            "level": level,
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Documentation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Documentation System (Step 44)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate documentation")
    gen_parser.add_argument("path", help="Module or package path")
    gen_parser.add_argument("--output", "-o", help="Output path")
    gen_parser.add_argument("--format", choices=["markdown", "html", "json"], default="markdown")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build all documentation")
    build_parser.add_argument("source", help="Source directory")
    build_parser.add_argument("--output", "-o", help="Output directory")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run documentation demo")

    args = parser.parse_args()

    docs = DocumentationManager()

    if args.command == "generate":
        path = Path(args.path)

        if path.is_dir():
            module_docs = docs.document_package(str(path))
            print(f"Documented {len(module_docs)} modules")
        else:
            module_doc = docs.document_module(str(path))
            content = docs.generate(module_doc)

            if args.output:
                docs.export(module_doc, args.output)
                print(f"Documentation written to {args.output}")
            else:
                print(content)

    elif args.command == "build":
        results = docs.build_all(args.source, args.output)
        print(f"Built documentation for {len(results)} files")
        for name, path in results.items():
            print(f"  {name}: {path}")

    elif args.command == "stats":
        stats = docs.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Documentation Statistics:")
            print(f"  Modules: {stats['modules_documented']}")
            print(f"  Classes: {stats['classes_documented']}")
            print(f"  Functions: {stats['functions_documented']}")
            print(f"  Endpoints: {stats['endpoints_documented']}")

    elif args.command == "demo":
        print("Running documentation demo...\n")

        # Document this file
        module_doc = docs.document_module(__file__)

        print(f"Module: {module_doc.name}")
        print(f"Description: {module_doc.description[:100]}...")
        print(f"Classes: {len(module_doc.classes)}")
        print(f"Functions: {len(module_doc.functions)}")

        if module_doc.classes:
            print(f"\nFirst class: {module_doc.classes[0].name}")
            print(f"  Methods: {len(module_doc.classes[0].methods)}")

        # Generate markdown
        print("\n--- Generated Markdown (first 500 chars) ---\n")
        markdown = docs.generate(module_doc, DocFormat.MARKDOWN)
        print(markdown[:500])

        print("\nDemo complete.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
