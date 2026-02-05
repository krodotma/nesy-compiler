#!/usr/bin/env python3
"""
documentation_module.py - Documentation Module (Step 94)

PBTSO Phase: SKILL, ITERATE

Provides:
- API documentation generation
- User guide generation
- Code documentation extraction
- Multiple output formats (Markdown, HTML, JSON)
- Documentation indexing

Bus Topics:
- code.docs.generate
- code.docs.index
- code.docs.export

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Type, Union

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
    HTML = "html"
    JSON = "json"
    RST = "rst"


class DocType(Enum):
    """Documentation types."""
    API = "api"
    GUIDE = "guide"
    REFERENCE = "reference"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"


@dataclass
class DocConfig:
    """Configuration for documentation module."""
    output_dir: str = "/pluribus/docs/generated"
    default_format: DocFormat = DocFormat.MARKDOWN
    include_private: bool = False
    include_source: bool = True
    max_examples: int = 3
    generate_toc: bool = True
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "default_format": self.default_format.value,
            "include_private": self.include_private,
            "include_source": self.include_source,
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
# Documentation Types
# =============================================================================

@dataclass
class DocParameter:
    """Documentation for a parameter."""
    name: str
    type: str
    description: str
    default: Optional[str] = None
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "default": self.default,
            "required": self.required,
        }


@dataclass
class DocExample:
    """Documentation example."""
    title: str
    code: str
    language: str = "python"
    description: str = ""
    output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "code": self.code,
            "language": self.language,
            "description": self.description,
            "output": self.output,
        }


@dataclass
class DocSection:
    """A section of documentation."""
    title: str
    content: str
    level: int = 1
    subsections: List["DocSection"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "subsections": [s.to_dict() for s in self.subsections],
        }


@dataclass
class FunctionDoc:
    """Documentation for a function/method."""
    name: str
    signature: str
    description: str
    parameters: List[DocParameter] = field(default_factory=list)
    returns: Optional[str] = None
    return_type: Optional[str] = None
    raises: List[str] = field(default_factory=list)
    examples: List[DocExample] = field(default_factory=list)
    deprecated: bool = False
    since: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "returns": self.returns,
            "return_type": self.return_type,
            "raises": self.raises,
            "examples": [e.to_dict() for e in self.examples],
            "deprecated": self.deprecated,
        }


@dataclass
class ClassDoc:
    """Documentation for a class."""
    name: str
    description: str
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionDoc] = field(default_factory=list)
    attributes: List[DocParameter] = field(default_factory=list)
    examples: List[DocExample] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "attributes": [a.to_dict() for a in self.attributes],
            "examples": [e.to_dict() for e in self.examples],
        }


@dataclass
class ModuleDoc:
    """Documentation for a module."""
    name: str
    path: str
    description: str
    classes: List[ClassDoc] = field(default_factory=list)
    functions: List[FunctionDoc] = field(default_factory=list)
    constants: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "constants": self.constants,
        }


@dataclass
class APIDoc:
    """Complete API documentation."""
    title: str
    version: str
    description: str
    modules: List[ModuleDoc] = field(default_factory=list)
    sections: List[DocSection] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "version": self.version,
            "description": self.description,
            "modules": [m.to_dict() for m in self.modules],
            "sections": [s.to_dict() for s in self.sections],
            "generated_at": self.generated_at,
        }


# =============================================================================
# Documentation Extractor
# =============================================================================

class DocExtractor:
    """Extract documentation from Python code."""

    def __init__(self, include_private: bool = False):
        self.include_private = include_private

    def extract_module(self, module_path: Path) -> Optional[ModuleDoc]:
        """Extract documentation from a Python module file."""
        try:
            source = module_path.read_text()
            tree = ast.parse(source)
        except Exception:
            return None

        # Get module docstring
        module_doc = ast.get_docstring(tree) or ""

        module = ModuleDoc(
            name=module_path.stem,
            path=str(module_path),
            description=module_doc,
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if self._should_include(node.name):
                    class_doc = self._extract_class(node)
                    module.classes.append(class_doc)

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if self._should_include(node.name) and not self._is_method(node, tree):
                    func_doc = self._extract_function(node)
                    module.functions.append(func_doc)

        return module

    def _should_include(self, name: str) -> bool:
        """Check if name should be included."""
        if name.startswith("_") and not self.include_private:
            return False
        return True

    def _is_method(self, node: ast.FunctionDef, tree: ast.Module) -> bool:
        """Check if function is a method of a class."""
        for other in ast.walk(tree):
            if isinstance(other, ast.ClassDef):
                for item in other.body:
                    if item is node:
                        return True
        return False

    def _extract_class(self, node: ast.ClassDef) -> ClassDoc:
        """Extract class documentation."""
        doc = ClassDoc(
            name=node.name,
            description=ast.get_docstring(node) or "",
            bases=[self._get_base_name(b) for b in node.bases],
        )

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._should_include(item.name):
                    method_doc = self._extract_function(item)
                    doc.methods.append(method_doc)

        return doc

    def _get_base_name(self, node: ast.expr) -> str:
        """Get base class name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_base_name(node.value)}.{node.attr}"
        return "?"

    def _extract_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionDoc:
        """Extract function documentation."""
        docstring = ast.get_docstring(node) or ""

        # Parse signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        signature = f"{node.name}({', '.join(args)})"
        if isinstance(node, ast.AsyncFunctionDef):
            signature = f"async {signature}"

        # Parse docstring for parameters
        parameters = self._parse_docstring_params(docstring, node.args)

        # Get return type
        return_type = None
        if node.returns:
            return_type = self._get_annotation_name(node.returns)

        # Parse docstring sections
        returns_doc = self._parse_docstring_section(docstring, "Returns")
        raises = self._parse_docstring_list(docstring, "Raises")

        return FunctionDoc(
            name=node.name,
            signature=signature,
            description=self._get_description(docstring),
            parameters=parameters,
            returns=returns_doc,
            return_type=return_type,
            raises=raises,
        )

    def _get_annotation_name(self, node: ast.expr) -> str:
        """Get annotation type name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_name(node.value)
            slice_val = self._get_annotation_name(node.slice)
            return f"{value}[{slice_val}]"
        return "Any"

    def _get_description(self, docstring: str) -> str:
        """Get description from docstring (first paragraph)."""
        if not docstring:
            return ""

        lines = docstring.split("\n\n")
        return lines[0].strip()

    def _parse_docstring_params(self, docstring: str, args: ast.arguments) -> List[DocParameter]:
        """Parse parameters from docstring."""
        params = []

        # Build param info from AST
        defaults = [None] * (len(args.args) - len(args.defaults)) + args.defaults

        for i, arg in enumerate(args.args):
            if arg.arg == "self":
                continue

            param = DocParameter(
                name=arg.arg,
                type=self._get_annotation_name(arg.annotation) if arg.annotation else "Any",
                description="",
                default=None if defaults[i] is None else ast.unparse(defaults[i]),
                required=defaults[i] is None,
            )

            # Try to find description in docstring
            pattern = rf"{arg.arg}\s*[:\-]\s*(.*?)(?:\n\s*\n|\n\s*[A-Z]|\Z)"
            match = re.search(pattern, docstring, re.DOTALL)
            if match:
                param.description = match.group(1).strip()

            params.append(param)

        return params

    def _parse_docstring_section(self, docstring: str, section: str) -> Optional[str]:
        """Parse a section from docstring."""
        pattern = rf"{section}:\s*\n(.*?)(?:\n\s*\n|\n[A-Z]|\Z)"
        match = re.search(pattern, docstring, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _parse_docstring_list(self, docstring: str, section: str) -> List[str]:
        """Parse a list section from docstring."""
        content = self._parse_docstring_section(docstring, section)
        if not content:
            return []

        items = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                line = line[1:].strip()
            if line:
                items.append(line)

        return items


# =============================================================================
# Documentation Generator
# =============================================================================

class DocGenerator:
    """Generate documentation in various formats."""

    def generate(self, doc: APIDoc, format: DocFormat) -> str:
        """Generate documentation in specified format."""
        if format == DocFormat.MARKDOWN:
            return self._generate_markdown(doc)
        elif format == DocFormat.HTML:
            return self._generate_html(doc)
        elif format == DocFormat.JSON:
            return json.dumps(doc.to_dict(), indent=2)
        elif format == DocFormat.RST:
            return self._generate_rst(doc)
        else:
            return json.dumps(doc.to_dict(), indent=2)

    def _generate_markdown(self, doc: APIDoc) -> str:
        """Generate Markdown documentation."""
        lines = []

        # Title
        lines.append(f"# {doc.title}")
        lines.append("")
        lines.append(f"**Version:** {doc.version}")
        lines.append("")
        lines.append(doc.description)
        lines.append("")

        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for module in doc.modules:
            lines.append(f"- [{module.name}](#{module.name.lower().replace('.', '')})")
        lines.append("")

        # Modules
        for module in doc.modules:
            lines.extend(self._module_to_markdown(module))
            lines.append("")

        # Additional sections
        for section in doc.sections:
            lines.extend(self._section_to_markdown(section))

        return "\n".join(lines)

    def _module_to_markdown(self, module: ModuleDoc) -> List[str]:
        """Convert module to Markdown."""
        lines = []
        lines.append(f"## {module.name}")
        lines.append("")
        lines.append(module.description)
        lines.append("")

        # Classes
        for cls in module.classes:
            lines.extend(self._class_to_markdown(cls))

        # Functions
        if module.functions:
            lines.append("### Functions")
            lines.append("")
            for func in module.functions:
                lines.extend(self._function_to_markdown(func))

        return lines

    def _class_to_markdown(self, cls: ClassDoc) -> List[str]:
        """Convert class to Markdown."""
        lines = []
        lines.append(f"### class `{cls.name}`")
        lines.append("")

        if cls.bases:
            lines.append(f"**Inherits from:** {', '.join(cls.bases)}")
            lines.append("")

        lines.append(cls.description)
        lines.append("")

        # Methods
        if cls.methods:
            lines.append("#### Methods")
            lines.append("")
            for method in cls.methods:
                lines.extend(self._function_to_markdown(method, indent="##### "))

        return lines

    def _function_to_markdown(self, func: FunctionDoc, indent: str = "#### ") -> List[str]:
        """Convert function to Markdown."""
        lines = []
        lines.append(f"{indent}`{func.signature}`")
        lines.append("")

        if func.deprecated:
            lines.append("**DEPRECATED**")
            lines.append("")

        lines.append(func.description)
        lines.append("")

        # Parameters
        if func.parameters:
            lines.append("**Parameters:**")
            lines.append("")
            for param in func.parameters:
                req = "" if param.required else " (optional)"
                default = f" = {param.default}" if param.default else ""
                lines.append(f"- `{param.name}` ({param.type}{default}){req}: {param.description}")
            lines.append("")

        # Returns
        if func.returns or func.return_type:
            ret_type = f" ({func.return_type})" if func.return_type else ""
            lines.append(f"**Returns:**{ret_type} {func.returns or ''}")
            lines.append("")

        # Raises
        if func.raises:
            lines.append("**Raises:**")
            for r in func.raises:
                lines.append(f"- {r}")
            lines.append("")

        # Examples
        if func.examples:
            lines.append("**Example:**")
            for ex in func.examples:
                lines.append(f"```{ex.language}")
                lines.append(ex.code)
                lines.append("```")
            lines.append("")

        return lines

    def _section_to_markdown(self, section: DocSection) -> List[str]:
        """Convert section to Markdown."""
        lines = []
        prefix = "#" * (section.level + 1)
        lines.append(f"{prefix} {section.title}")
        lines.append("")
        lines.append(section.content)
        lines.append("")

        for subsection in section.subsections:
            lines.extend(self._section_to_markdown(subsection))

        return lines

    def _generate_html(self, doc: APIDoc) -> str:
        """Generate HTML documentation."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{doc.title}</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; overflow-x: auto; }}
        .deprecated {{ color: #dc3545; }}
        .param-name {{ font-weight: bold; }}
        .param-type {{ color: #666; }}
    </style>
</head>
<body>
    <h1>{doc.title}</h1>
    <p><strong>Version:</strong> {doc.version}</p>
    <p>{doc.description}</p>
"""

        for module in doc.modules:
            html += f"<h2>{module.name}</h2>\n"
            html += f"<p>{module.description}</p>\n"

            for cls in module.classes:
                html += f"<h3>class {cls.name}</h3>\n"
                html += f"<p>{cls.description}</p>\n"

                for method in cls.methods:
                    html += f"<h4><code>{method.signature}</code></h4>\n"
                    html += f"<p>{method.description}</p>\n"

        html += """
</body>
</html>"""
        return html

    def _generate_rst(self, doc: APIDoc) -> str:
        """Generate reStructuredText documentation."""
        lines = []
        lines.append("=" * len(doc.title))
        lines.append(doc.title)
        lines.append("=" * len(doc.title))
        lines.append("")
        lines.append(f"**Version:** {doc.version}")
        lines.append("")
        lines.append(doc.description)
        lines.append("")

        for module in doc.modules:
            lines.append(module.name)
            lines.append("-" * len(module.name))
            lines.append("")
            lines.append(module.description)
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Documentation Module
# =============================================================================

class DocumentationModule:
    """
    Documentation module for API docs and guides.

    PBTSO Phase: SKILL, ITERATE

    Features:
    - Extract docs from Python code
    - Generate multiple formats
    - Build API reference
    - Create user guides

    Usage:
        docs = DocumentationModule()
        api_doc = docs.generate_api_doc("/path/to/module.py")
        markdown = docs.export(api_doc, DocFormat.MARKDOWN)
    """

    BUS_TOPICS = {
        "generate": "code.docs.generate",
        "index": "code.docs.index",
        "export": "code.docs.export",
    }

    def __init__(
        self,
        config: Optional[DocConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or DocConfig()
        self.bus = bus or LockedAgentBus()

        self._extractor = DocExtractor(include_private=self.config.include_private)
        self._generator = DocGenerator()

        self._docs: Dict[str, APIDoc] = {}
        self._lock = Lock()

    # =========================================================================
    # Documentation Generation
    # =========================================================================

    def generate_module_doc(self, module_path: Union[str, Path]) -> Optional[ModuleDoc]:
        """Generate documentation for a module."""
        path = Path(module_path)
        return self._extractor.extract_module(path)

    def generate_api_doc(
        self,
        paths: List[Union[str, Path]],
        title: str = "API Documentation",
        version: str = "1.0.0",
        description: str = "",
    ) -> APIDoc:
        """Generate API documentation from multiple modules."""
        modules = []

        for path in paths:
            module_doc = self.generate_module_doc(path)
            if module_doc:
                modules.append(module_doc)

        api_doc = APIDoc(
            title=title,
            version=version,
            description=description,
            modules=modules,
        )

        self.bus.emit({
            "topic": self.BUS_TOPICS["generate"],
            "kind": "docs",
            "actor": "documentation-module",
            "data": {
                "title": title,
                "module_count": len(modules),
            },
        })

        with self._lock:
            self._docs[title] = api_doc

        return api_doc

    def generate_from_class(self, cls: Type) -> ClassDoc:
        """Generate documentation from a Python class."""
        return ClassDoc(
            name=cls.__name__,
            description=inspect.getdoc(cls) or "",
            bases=[b.__name__ for b in cls.__bases__ if b is not object],
            methods=[
                self._extract_method_doc(name, method)
                for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
                if not name.startswith("_") or name == "__init__"
            ],
        )

    def _extract_method_doc(self, name: str, method: Callable) -> FunctionDoc:
        """Extract documentation from a method."""
        sig = inspect.signature(method)

        params = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            annotation = param.annotation
            type_str = annotation.__name__ if annotation != inspect.Parameter.empty else "Any"

            default = None
            if param.default != inspect.Parameter.empty:
                default = repr(param.default)

            params.append(DocParameter(
                name=param_name,
                type=type_str,
                description="",
                default=default,
                required=param.default == inspect.Parameter.empty,
            ))

        return FunctionDoc(
            name=name,
            signature=f"{name}{sig}",
            description=inspect.getdoc(method) or "",
            parameters=params,
        )

    # =========================================================================
    # Export
    # =========================================================================

    def export(
        self,
        doc: APIDoc,
        format: Optional[DocFormat] = None,
    ) -> str:
        """Export documentation to string."""
        format = format or self.config.default_format
        content = self._generator.generate(doc, format)

        self.bus.emit({
            "topic": self.BUS_TOPICS["export"],
            "kind": "docs",
            "actor": "documentation-module",
            "data": {
                "title": doc.title,
                "format": format.value,
                "size": len(content),
            },
        })

        return content

    def export_to_file(
        self,
        doc: APIDoc,
        output_path: Union[str, Path],
        format: Optional[DocFormat] = None,
    ) -> bool:
        """Export documentation to file."""
        content = self.export(doc, format)

        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return True
        except Exception:
            return False

    # =========================================================================
    # Guides
    # =========================================================================

    def create_guide(
        self,
        title: str,
        sections: List[DocSection],
        version: str = "1.0.0",
    ) -> APIDoc:
        """Create a user guide."""
        return APIDoc(
            title=title,
            version=version,
            description="",
            sections=sections,
        )

    def create_section(
        self,
        title: str,
        content: str,
        level: int = 1,
        subsections: Optional[List[DocSection]] = None,
    ) -> DocSection:
        """Create a documentation section."""
        return DocSection(
            title=title,
            content=content,
            level=level,
            subsections=subsections or [],
        )

    # =========================================================================
    # Utilities
    # =========================================================================

    def list_docs(self) -> List[str]:
        """List generated documentation."""
        return list(self._docs.keys())

    def get_doc(self, title: str) -> Optional[APIDoc]:
        """Get generated documentation by title."""
        return self._docs.get(title)

    def stats(self) -> Dict[str, Any]:
        """Get documentation statistics."""
        return {
            "total_docs": len(self._docs),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Documentation Module."""
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Module (Step 94)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate documentation")
    gen_parser.add_argument("paths", nargs="+", help="Python files to document")
    gen_parser.add_argument("--title", "-t", default="API Documentation")
    gen_parser.add_argument("--version", "-v", default="1.0.0")
    gen_parser.add_argument("--format", "-f", choices=["markdown", "html", "json"], default="markdown")
    gen_parser.add_argument("--output", "-o", help="Output file")

    # extract command
    ext_parser = subparsers.add_parser("extract", help="Extract module info")
    ext_parser.add_argument("path", help="Python file")
    ext_parser.add_argument("--json", action="store_true")

    # demo command
    subparsers.add_parser("demo", help="Run documentation demo")

    args = parser.parse_args()
    docs = DocumentationModule()

    if args.command == "generate":
        api_doc = docs.generate_api_doc(
            args.paths,
            title=args.title,
            version=args.version,
        )

        format_map = {
            "markdown": DocFormat.MARKDOWN,
            "html": DocFormat.HTML,
            "json": DocFormat.JSON,
        }
        format = format_map[args.format]

        content = docs.export(api_doc, format)

        if args.output:
            Path(args.output).write_text(content)
            print(f"Documentation written to {args.output}")
        else:
            print(content)

        return 0

    elif args.command == "extract":
        module_doc = docs.generate_module_doc(args.path)
        if module_doc:
            if args.json:
                print(json.dumps(module_doc.to_dict(), indent=2))
            else:
                print(f"Module: {module_doc.name}")
                print(f"Description: {module_doc.description[:100]}...")
                print(f"Classes: {len(module_doc.classes)}")
                print(f"Functions: {len(module_doc.functions)}")

                for cls in module_doc.classes:
                    print(f"\n  Class: {cls.name}")
                    print(f"    Methods: {len(cls.methods)}")
        return 0

    elif args.command == "demo":
        print("Documentation Module Demo\n")

        # Generate docs for this module
        api_doc = docs.generate_api_doc(
            [Path(__file__)],
            title="Documentation Module API",
            version="0.1.0",
            description="API documentation for the Documentation Module.",
        )

        print("Generated API Documentation:")
        print(f"  Modules: {len(api_doc.modules)}")
        for mod in api_doc.modules:
            print(f"    - {mod.name}: {len(mod.classes)} classes, {len(mod.functions)} functions")

        print("\nMarkdown Preview (first 500 chars):")
        md = docs.export(api_doc, DocFormat.MARKDOWN)
        print(md[:500] + "...\n")

        # Create a guide
        guide = docs.create_guide(
            "Getting Started Guide",
            sections=[
                docs.create_section("Introduction", "Welcome to the Code Agent documentation system."),
                docs.create_section("Installation", "Install using pip: `pip install code-agent`"),
                docs.create_section("Usage", "Import and use the module as shown below.", subsections=[
                    docs.create_section("Basic Usage", "```python\nfrom code.docs import DocumentationModule\ndocs = DocumentationModule()\n```", level=2),
                ]),
            ],
        )

        print("Created User Guide:")
        print(docs.export(guide, DocFormat.MARKDOWN)[:400] + "...")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
