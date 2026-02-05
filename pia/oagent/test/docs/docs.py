#!/usr/bin/env python3
"""
Step 144: Test Documentation Module

API documentation and guide generation for the Test Agent.

PBTSO Phase: OBSERVE, DISTRIBUTE
Bus Topics:
- test.docs.generate (emits)
- test.docs.validate (emits)
- test.docs.publish (emits)

Dependencies: Steps 101-143 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import inspect
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type


# ============================================================================
# Constants
# ============================================================================

class DocFormat(Enum):
    """Documentation output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    OPENAPI = "openapi"


class DocType(Enum):
    """Types of documentation."""
    API = "api"
    GUIDE = "guide"
    REFERENCE = "reference"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class Parameter:
    """
    API parameter documentation.

    Attributes:
        name: Parameter name
        param_type: Parameter type
        description: Parameter description
        required: Whether parameter is required
        default: Default value
        example: Example value
    """
    name: str
    param_type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    example: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.param_type,
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "example": self.example,
        }


@dataclass
class APIEndpoint:
    """
    API endpoint documentation.

    Attributes:
        path: Endpoint path
        method: HTTP method
        description: Endpoint description
        parameters: List of parameters
        request_body: Request body schema
        responses: Response schemas
        examples: Usage examples
        tags: Endpoint tags
    """
    path: str
    method: str = "GET"
    description: str = ""
    summary: str = ""
    parameters: List[Parameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "summary": self.summary,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "requestBody": self.request_body,
            "responses": self.responses,
            "examples": self.examples,
            "tags": self.tags,
            "deprecated": self.deprecated,
        }


@dataclass
class APIDoc:
    """
    Complete API documentation.

    Attributes:
        title: API title
        version: API version
        description: API description
        base_url: Base URL
        endpoints: List of endpoints
        schemas: Schema definitions
        security: Security schemes
    """
    title: str
    version: str = "1.0.0"
    description: str = ""
    base_url: str = "/api/v1"
    endpoints: List[APIEndpoint] = field(default_factory=list)
    schemas: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)

    def add_endpoint(self, endpoint: APIEndpoint) -> None:
        """Add an endpoint."""
        self.endpoints.append(endpoint)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "version": self.version,
            "description": self.description,
            "base_url": self.base_url,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "schemas": self.schemas,
            "security": self.security,
        }

    def to_openapi(self) -> Dict[str, Any]:
        """Convert to OpenAPI format."""
        paths = {}
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}

            paths[endpoint.path][endpoint.method.lower()] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "parameters": [p.to_dict() for p in endpoint.parameters],
                "responses": endpoint.responses or {"200": {"description": "Success"}},
                "deprecated": endpoint.deprecated,
            }
            if endpoint.request_body:
                paths[endpoint.path][endpoint.method.lower()]["requestBody"] = endpoint.request_body

        return {
            "openapi": "3.0.3",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
            },
            "servers": [{"url": self.base_url}],
            "paths": paths,
            "components": {
                "schemas": self.schemas,
                "securitySchemes": self.security,
            },
        }


@dataclass
class GuideSection:
    """
    A section in a guide.

    Attributes:
        title: Section title
        content: Section content
        code_examples: Code examples
        subsections: Nested sections
    """
    title: str
    content: str = ""
    code_examples: List[Dict[str, str]] = field(default_factory=list)
    subsections: List["GuideSection"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "code_examples": self.code_examples,
            "subsections": [s.to_dict() for s in self.subsections],
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert to markdown."""
        lines = [f"{'#' * level} {self.title}", "", self.content, ""]

        for example in self.code_examples:
            lang = example.get("language", "python")
            code = example.get("code", "")
            lines.extend([f"```{lang}", code, "```", ""])

        for subsection in self.subsections:
            lines.append(subsection.to_markdown(level + 1))

        return "\n".join(lines)


@dataclass
class GuideDoc:
    """
    A guide document.

    Attributes:
        title: Guide title
        description: Guide description
        sections: Guide sections
        prerequisites: Prerequisites
        tags: Guide tags
    """
    title: str
    description: str = ""
    sections: List[GuideSection] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_section(self, section: GuideSection) -> None:
        """Add a section."""
        self.sections.append(section)
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "sections": [s.to_dict() for s in self.sections],
            "prerequisites": self.prerequisites,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_markdown(self) -> str:
        """Convert to markdown."""
        lines = [f"# {self.title}", "", self.description, ""]

        if self.prerequisites:
            lines.append("## Prerequisites")
            lines.append("")
            for prereq in self.prerequisites:
                lines.append(f"- {prereq}")
            lines.append("")

        for section in self.sections:
            lines.append(section.to_markdown())

        return "\n".join(lines)


@dataclass
class ChangelogEntry:
    """
    A changelog entry.

    Attributes:
        version: Version number
        date: Release date
        added: Added features
        changed: Changed features
        deprecated: Deprecated features
        removed: Removed features
        fixed: Bug fixes
        security: Security fixes
    """
    version: str
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    added: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    deprecated: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    fixed: List[str] = field(default_factory=list)
    security: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "date": self.date,
            "added": self.added,
            "changed": self.changed,
            "deprecated": self.deprecated,
            "removed": self.removed,
            "fixed": self.fixed,
            "security": self.security,
        }

    def to_markdown(self) -> str:
        """Convert to markdown."""
        lines = [f"## [{self.version}] - {self.date}", ""]

        sections = [
            ("Added", self.added),
            ("Changed", self.changed),
            ("Deprecated", self.deprecated),
            ("Removed", self.removed),
            ("Fixed", self.fixed),
            ("Security", self.security),
        ]

        for title, items in sections:
            if items:
                lines.append(f"### {title}")
                lines.append("")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")

        return "\n".join(lines)


@dataclass
class DocsConfig:
    """
    Configuration for documentation generation.

    Attributes:
        output_dir: Output directory
        default_format: Default output format
        include_examples: Include examples
        include_schemas: Include schema definitions
        auto_generate: Auto-generate from code
    """
    output_dir: str = ".pluribus/test-agent/docs"
    default_format: DocFormat = DocFormat.MARKDOWN
    include_examples: bool = True
    include_schemas: bool = True
    auto_generate: bool = True
    template_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": self.output_dir,
            "default_format": self.default_format.value,
            "include_examples": self.include_examples,
            "include_schemas": self.include_schemas,
            "auto_generate": self.auto_generate,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class DocsBus:
    """Bus interface for docs with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass


# ============================================================================
# Test Doc Generator
# ============================================================================

class TestDocGenerator:
    """
    Documentation generator for the Test Agent.

    Features:
    - API documentation generation
    - Guide generation
    - Changelog management
    - OpenAPI spec generation
    - Auto-generation from code

    PBTSO Phase: OBSERVE, DISTRIBUTE
    Bus Topics: test.docs.generate, test.docs.validate, test.docs.publish
    """

    BUS_TOPICS = {
        "generate": "test.docs.generate",
        "validate": "test.docs.validate",
        "publish": "test.docs.publish",
    }

    def __init__(self, bus=None, config: Optional[DocsConfig] = None):
        """
        Initialize the doc generator.

        Args:
            bus: Optional bus instance
            config: Documentation configuration
        """
        self.bus = bus or DocsBus()
        self.config = config or DocsConfig()
        self._api_docs: Dict[str, APIDoc] = {}
        self._guides: Dict[str, GuideDoc] = {}
        self._changelog: List[ChangelogEntry] = []

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def create_api_doc(
        self,
        name: str,
        title: str,
        version: str = "1.0.0",
        description: str = "",
        base_url: str = "/api/v1",
    ) -> APIDoc:
        """
        Create a new API documentation.

        Args:
            name: Doc identifier
            title: API title
            version: API version
            description: API description
            base_url: Base URL

        Returns:
            APIDoc instance
        """
        doc = APIDoc(
            title=title,
            version=version,
            description=description,
            base_url=base_url,
        )
        self._api_docs[name] = doc
        return doc

    def add_endpoint(
        self,
        doc_name: str,
        path: str,
        method: str = "GET",
        description: str = "",
        parameters: Optional[List[Parameter]] = None,
        responses: Optional[Dict[str, Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
    ) -> APIEndpoint:
        """
        Add an endpoint to an API doc.

        Args:
            doc_name: API doc name
            path: Endpoint path
            method: HTTP method
            description: Endpoint description
            parameters: Endpoint parameters
            responses: Response schemas
            tags: Endpoint tags

        Returns:
            APIEndpoint instance
        """
        if doc_name not in self._api_docs:
            raise ValueError(f"API doc not found: {doc_name}")

        endpoint = APIEndpoint(
            path=path,
            method=method,
            description=description,
            parameters=parameters or [],
            responses=responses or {},
            tags=tags or [],
        )
        self._api_docs[doc_name].add_endpoint(endpoint)
        return endpoint

    def document_class(self, cls: Type) -> APIDoc:
        """
        Auto-generate API documentation from a class.

        Args:
            cls: Class to document

        Returns:
            APIDoc instance
        """
        doc = APIDoc(
            title=cls.__name__,
            description=inspect.getdoc(cls) or "",
        )

        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue

            docstring = inspect.getdoc(method) or ""
            sig = inspect.signature(method)

            params = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                params.append(Parameter(
                    name=param_name,
                    param_type=str(param.annotation) if param.annotation != inspect.Parameter.empty else "any",
                    required=param.default == inspect.Parameter.empty,
                    default=None if param.default == inspect.Parameter.empty else param.default,
                ))

            endpoint = APIEndpoint(
                path=f"/{name}",
                method="POST" if params else "GET",
                description=docstring,
                parameters=params,
            )
            doc.add_endpoint(endpoint)

        return doc

    def create_guide(
        self,
        name: str,
        title: str,
        description: str = "",
        prerequisites: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> GuideDoc:
        """
        Create a new guide document.

        Args:
            name: Guide identifier
            title: Guide title
            description: Guide description
            prerequisites: Prerequisites
            tags: Guide tags

        Returns:
            GuideDoc instance
        """
        guide = GuideDoc(
            title=title,
            description=description,
            prerequisites=prerequisites or [],
            tags=tags or [],
        )
        self._guides[name] = guide
        return guide

    def add_changelog_entry(
        self,
        version: str,
        added: Optional[List[str]] = None,
        changed: Optional[List[str]] = None,
        deprecated: Optional[List[str]] = None,
        removed: Optional[List[str]] = None,
        fixed: Optional[List[str]] = None,
        security: Optional[List[str]] = None,
    ) -> ChangelogEntry:
        """
        Add a changelog entry.

        Args:
            version: Version number
            added: Added features
            changed: Changed features
            deprecated: Deprecated features
            removed: Removed features
            fixed: Bug fixes
            security: Security fixes

        Returns:
            ChangelogEntry instance
        """
        entry = ChangelogEntry(
            version=version,
            added=added or [],
            changed=changed or [],
            deprecated=deprecated or [],
            removed=removed or [],
            fixed=fixed or [],
            security=security or [],
        )
        self._changelog.insert(0, entry)  # Newest first
        return entry

    def generate(
        self,
        doc_type: DocType,
        name: str,
        output_format: Optional[DocFormat] = None,
    ) -> str:
        """
        Generate documentation.

        Args:
            doc_type: Type of documentation
            name: Doc name
            output_format: Output format

        Returns:
            Generated documentation content
        """
        fmt = output_format or self.config.default_format
        content = ""

        self._emit_event("generate", {
            "doc_type": doc_type.value,
            "name": name,
            "format": fmt.value,
        })

        if doc_type == DocType.API:
            if name not in self._api_docs:
                raise ValueError(f"API doc not found: {name}")
            content = self._generate_api_doc(self._api_docs[name], fmt)

        elif doc_type == DocType.GUIDE:
            if name not in self._guides:
                raise ValueError(f"Guide not found: {name}")
            content = self._generate_guide(self._guides[name], fmt)

        elif doc_type == DocType.CHANGELOG:
            content = self._generate_changelog(fmt)

        return content

    def _generate_api_doc(self, doc: APIDoc, fmt: DocFormat) -> str:
        """Generate API documentation."""
        if fmt == DocFormat.OPENAPI:
            return json.dumps(doc.to_openapi(), indent=2)
        elif fmt == DocFormat.JSON:
            return json.dumps(doc.to_dict(), indent=2)
        elif fmt == DocFormat.MARKDOWN:
            return self._api_to_markdown(doc)
        elif fmt == DocFormat.HTML:
            md = self._api_to_markdown(doc)
            return self._markdown_to_html(md)
        return ""

    def _api_to_markdown(self, doc: APIDoc) -> str:
        """Convert API doc to markdown."""
        lines = [
            f"# {doc.title}",
            "",
            f"**Version:** {doc.version}",
            "",
            doc.description,
            "",
            f"**Base URL:** `{doc.base_url}`",
            "",
            "## Endpoints",
            "",
        ]

        for endpoint in doc.endpoints:
            deprecated = " (Deprecated)" if endpoint.deprecated else ""
            lines.append(f"### {endpoint.method} {endpoint.path}{deprecated}")
            lines.append("")
            lines.append(endpoint.description)
            lines.append("")

            if endpoint.parameters:
                lines.append("**Parameters:**")
                lines.append("")
                lines.append("| Name | Type | Required | Description |")
                lines.append("|------|------|----------|-------------|")
                for param in endpoint.parameters:
                    required = "Yes" if param.required else "No"
                    lines.append(f"| {param.name} | {param.param_type} | {required} | {param.description} |")
                lines.append("")

            if endpoint.responses:
                lines.append("**Responses:**")
                lines.append("")
                for code, response in endpoint.responses.items():
                    lines.append(f"- `{code}`: {response.get('description', '')}")
                lines.append("")

        return "\n".join(lines)

    def _generate_guide(self, guide: GuideDoc, fmt: DocFormat) -> str:
        """Generate guide documentation."""
        if fmt == DocFormat.JSON:
            return json.dumps(guide.to_dict(), indent=2)
        elif fmt == DocFormat.MARKDOWN:
            return guide.to_markdown()
        elif fmt == DocFormat.HTML:
            return self._markdown_to_html(guide.to_markdown())
        return ""

    def _generate_changelog(self, fmt: DocFormat) -> str:
        """Generate changelog."""
        if fmt == DocFormat.JSON:
            return json.dumps([e.to_dict() for e in self._changelog], indent=2)
        elif fmt == DocFormat.MARKDOWN:
            lines = ["# Changelog", "", "All notable changes to this project.", ""]
            for entry in self._changelog:
                lines.append(entry.to_markdown())
            return "\n".join(lines)
        elif fmt == DocFormat.HTML:
            md = self._generate_changelog(DocFormat.MARKDOWN)
            return self._markdown_to_html(md)
        return ""

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML (basic conversion)."""
        html = markdown

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Code
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # Lists
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Paragraphs
        html = re.sub(r'\n\n', r'</p><p>', html)

        return f"<html><body><p>{html}</p></body></html>"

    def save(
        self,
        doc_type: DocType,
        name: str,
        output_format: Optional[DocFormat] = None,
    ) -> Path:
        """
        Save documentation to file.

        Args:
            doc_type: Type of documentation
            name: Doc name
            output_format: Output format

        Returns:
            Path to saved file
        """
        fmt = output_format or self.config.default_format
        content = self.generate(doc_type, name, fmt)

        ext = {
            DocFormat.MARKDOWN: ".md",
            DocFormat.HTML: ".html",
            DocFormat.JSON: ".json",
            DocFormat.OPENAPI: ".yaml",
        }.get(fmt, ".txt")

        output_path = Path(self.config.output_dir) / f"{name}{ext}"

        with open(output_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(content)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self._emit_event("publish", {
            "doc_type": doc_type.value,
            "name": name,
            "path": str(output_path),
        })

        return output_path

    def validate(self, doc_type: DocType, name: str) -> Dict[str, Any]:
        """
        Validate documentation.

        Args:
            doc_type: Type of documentation
            name: Doc name

        Returns:
            Validation result
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        if doc_type == DocType.API:
            if name not in self._api_docs:
                result["valid"] = False
                result["errors"].append(f"API doc not found: {name}")
            else:
                doc = self._api_docs[name]
                if not doc.title:
                    result["warnings"].append("Missing API title")
                if not doc.endpoints:
                    result["warnings"].append("No endpoints defined")
                for endpoint in doc.endpoints:
                    if not endpoint.description:
                        result["warnings"].append(f"Missing description for {endpoint.method} {endpoint.path}")

        elif doc_type == DocType.GUIDE:
            if name not in self._guides:
                result["valid"] = False
                result["errors"].append(f"Guide not found: {name}")
            else:
                guide = self._guides[name]
                if not guide.title:
                    result["warnings"].append("Missing guide title")
                if not guide.sections:
                    result["warnings"].append("No sections defined")

        self._emit_event("validate", {
            "doc_type": doc_type.value,
            "name": name,
            "valid": result["valid"],
        })

        return result

    def list_docs(self, doc_type: Optional[DocType] = None) -> Dict[str, List[str]]:
        """List all documentation."""
        result = {}

        if doc_type is None or doc_type == DocType.API:
            result["api"] = list(self._api_docs.keys())
        if doc_type is None or doc_type == DocType.GUIDE:
            result["guides"] = list(self._guides.keys())
        if doc_type is None or doc_type == DocType.CHANGELOG:
            result["changelog"] = [e.version for e in self._changelog]

        return result

    async def generate_async(
        self,
        doc_type: DocType,
        name: str,
        output_format: Optional[DocFormat] = None,
    ) -> str:
        """Async version of generate."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, doc_type, name, output_format
        )

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.docs.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "docs",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Doc Generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Doc Generator")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate documentation")
    gen_parser.add_argument("type", choices=["api", "guide", "changelog"])
    gen_parser.add_argument("name", help="Doc name")
    gen_parser.add_argument("--format", choices=["markdown", "html", "json", "openapi"],
                           default="markdown")

    # List command
    list_parser = subparsers.add_parser("list", help="List documentation")
    list_parser.add_argument("--type", choices=["api", "guide", "changelog"])

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate documentation")
    validate_parser.add_argument("type", choices=["api", "guide"])
    validate_parser.add_argument("name", help="Doc name")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/docs")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = DocsConfig(output_dir=args.output)
    generator = TestDocGenerator(config=config)

    # Create sample docs for testing
    api_doc = generator.create_api_doc(
        "test-agent",
        "Test Agent API",
        version="1.0.0",
        description="API for the Test Agent",
    )
    generator.add_endpoint(
        "test-agent",
        "/tests",
        "GET",
        "List all tests",
        tags=["tests"],
    )

    generator.create_guide(
        "quickstart",
        "Quick Start Guide",
        "Get started with the Test Agent",
        prerequisites=["Python 3.10+"],
    )

    generator.add_changelog_entry(
        "1.0.0",
        added=["Initial release", "Test generation", "Coverage analysis"],
    )

    if args.command == "generate":
        doc_type = DocType(args.type)
        fmt = DocFormat(args.format)

        try:
            content = generator.generate(doc_type, args.name, fmt)
            print(content)
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)

    elif args.command == "list":
        doc_type = DocType(args.type) if args.type else None
        docs = generator.list_docs(doc_type)

        if args.json:
            print(json.dumps(docs, indent=2))
        else:
            print("\nDocumentation:")
            for dtype, names in docs.items():
                print(f"\n  {dtype}:")
                for name in names:
                    print(f"    - {name}")

    elif args.command == "validate":
        doc_type = DocType(args.type)
        result = generator.validate(doc_type, args.name)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            status = "[VALID]" if result["valid"] else "[INVALID]"
            print(f"\n{status} {args.type}/{args.name}")
            if result["errors"]:
                print("\n  Errors:")
                for error in result["errors"]:
                    print(f"    - {error}")
            if result["warnings"]:
                print("\n  Warnings:")
                for warning in result["warnings"]:
                    print(f"    - {warning}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
