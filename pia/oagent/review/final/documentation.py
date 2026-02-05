#!/usr/bin/env python3
"""
Documentation Generator (Step 194)

API documentation and guide generation system for the Review Agent.
Supports multiple output formats and automatic API doc extraction.

PBTSO Phase: DISTILL
Bus Topics: review.docs.generate, review.docs.publish

Documentation Features:
- API documentation generation
- User guide generation
- Multiple output formats (Markdown, HTML, JSON)
- Docstring extraction
- Example generation

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import ast
import asyncio
import fcntl
import inspect
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class DocFormat(Enum):
    """Documentation output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    RST = "rst"  # ReStructuredText


class DocType(Enum):
    """Documentation types."""
    API = "api"
    GUIDE = "guide"
    REFERENCE = "reference"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"


@dataclass
class Parameter:
    """API parameter documentation."""
    name: str
    type: str
    description: str
    required: bool = False
    default: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Endpoint:
    """API endpoint documentation."""
    name: str
    method: str
    path: str
    description: str
    parameters: List[Parameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    response: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "method": self.method,
            "path": self.path,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "request_body": self.request_body,
            "response": self.response,
            "examples": self.examples,
            "tags": self.tags,
        }


@dataclass
class Section:
    """Documentation section."""
    title: str
    content: str
    level: int = 1
    subsections: List["Section"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "subsections": [s.to_dict() for s in self.subsections],
        }


@dataclass
class Document:
    """Complete documentation document."""
    title: str
    doc_type: DocType
    sections: List[Section] = field(default_factory=list)
    endpoints: List[Endpoint] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "doc_type": self.doc_type.value,
            "sections": [s.to_dict() for s in self.sections],
            "endpoints": [e.to_dict() for e in self.endpoints],
            "version": self.version,
            "created_at": self.created_at,
        }


# ============================================================================
# Docstring Parser
# ============================================================================

class DocstringParser:
    """Parse Python docstrings."""

    @staticmethod
    def parse(docstring: Optional[str]) -> Dict[str, Any]:
        """
        Parse a docstring into structured data.

        Args:
            docstring: Raw docstring

        Returns:
            Parsed docstring with description, params, returns, etc.
        """
        if not docstring:
            return {"description": "", "params": [], "returns": None, "raises": []}

        lines = docstring.strip().split("\n")
        result = {
            "description": "",
            "params": [],
            "returns": None,
            "raises": [],
            "examples": [],
        }

        current_section = "description"
        description_lines = []
        current_param = None

        for line in lines:
            stripped = line.strip()

            # Detect section headers
            if stripped.lower().startswith("args:") or stripped.lower().startswith("arguments:"):
                current_section = "args"
                continue
            elif stripped.lower().startswith("parameters:"):
                current_section = "args"
                continue
            elif stripped.lower().startswith("returns:"):
                current_section = "returns"
                continue
            elif stripped.lower().startswith("raises:"):
                current_section = "raises"
                continue
            elif stripped.lower().startswith("example:") or stripped.lower().startswith("examples:"):
                current_section = "examples"
                continue
            elif stripped.lower().startswith("attributes:"):
                current_section = "attributes"
                continue

            # Process based on section
            if current_section == "description":
                description_lines.append(stripped)

            elif current_section == "args":
                # Parse parameter: name: description or name (type): description
                if stripped and not stripped.startswith(" "):
                    # New parameter
                    if ":" in stripped:
                        parts = stripped.split(":", 1)
                        name_part = parts[0].strip()
                        desc = parts[1].strip() if len(parts) > 1 else ""

                        # Check for type in parentheses
                        param_type = "any"
                        if "(" in name_part and ")" in name_part:
                            name = name_part[:name_part.index("(")].strip()
                            param_type = name_part[name_part.index("(") + 1:name_part.index(")")].strip()
                        else:
                            name = name_part

                        current_param = {"name": name, "type": param_type, "description": desc}
                        result["params"].append(current_param)
                elif current_param and stripped:
                    # Continuation of description
                    current_param["description"] += " " + stripped

            elif current_section == "returns":
                if result["returns"] is None:
                    result["returns"] = stripped
                else:
                    result["returns"] += " " + stripped

        result["description"] = " ".join(description_lines).strip()
        return result


# ============================================================================
# API Documentation Builder
# ============================================================================

class APIDocBuilder:
    """
    Builds API documentation from code.

    Example:
        builder = APIDocBuilder()

        # Add endpoint
        builder.add_endpoint(Endpoint(
            name="Review Files",
            method="POST",
            path="/api/v1/review",
            description="Submit files for review",
            parameters=[
                Parameter("files", "array[string]", "File paths to review", required=True),
            ],
        ))

        # Generate docs
        doc = builder.build()
        markdown = builder.to_markdown(doc)
    """

    def __init__(self):
        """Initialize API doc builder."""
        self._endpoints: List[Endpoint] = []
        self._title = "API Documentation"
        self._version = "1.0.0"
        self._description = ""

    def set_info(
        self,
        title: str,
        version: str = "1.0.0",
        description: str = "",
    ) -> None:
        """Set API information."""
        self._title = title
        self._version = version
        self._description = description

    def add_endpoint(self, endpoint: Endpoint) -> None:
        """Add an API endpoint."""
        self._endpoints.append(endpoint)

    def extract_from_class(self, cls: Type) -> None:
        """Extract endpoints from a class with decorated methods."""
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if hasattr(method, "_api_endpoint"):
                endpoint_info = method._api_endpoint
                docstring = DocstringParser.parse(method.__doc__)

                endpoint = Endpoint(
                    name=endpoint_info.get("name", name),
                    method=endpoint_info.get("method", "GET"),
                    path=endpoint_info.get("path", f"/{name}"),
                    description=docstring["description"],
                    parameters=[
                        Parameter(
                            name=p["name"],
                            type=p["type"],
                            description=p["description"],
                        )
                        for p in docstring["params"]
                    ],
                    tags=endpoint_info.get("tags", []),
                )
                self._endpoints.append(endpoint)

    def build(self) -> Document:
        """Build the API documentation."""
        doc = Document(
            title=self._title,
            doc_type=DocType.API,
            version=self._version,
            endpoints=self._endpoints,
        )

        # Add overview section
        if self._description:
            doc.sections.append(Section(
                title="Overview",
                content=self._description,
                level=1,
            ))

        # Add endpoints section
        if self._endpoints:
            endpoints_section = Section(
                title="Endpoints",
                content="",
                level=1,
            )

            # Group by tag
            tags = set()
            for ep in self._endpoints:
                tags.update(ep.tags)

            for tag in sorted(tags) or [""]:
                tag_endpoints = [ep for ep in self._endpoints if tag in ep.tags or not tag]
                if tag:
                    endpoints_section.subsections.append(Section(
                        title=tag.title(),
                        content="",
                        level=2,
                    ))

            doc.sections.append(endpoints_section)

        return doc

    def to_markdown(self, doc: Document) -> str:
        """Convert document to Markdown."""
        lines = []

        # Title
        lines.append(f"# {doc.title}")
        lines.append(f"\n**Version:** {doc.version}")
        lines.append(f"\n**Generated:** {doc.created_at}")
        lines.append("")

        # Sections
        for section in doc.sections:
            lines.append(f"\n{'#' * section.level} {section.title}\n")
            if section.content:
                lines.append(section.content)
                lines.append("")

        # Endpoints
        if doc.endpoints:
            lines.append("\n## Endpoints\n")
            for endpoint in doc.endpoints:
                lines.append(f"### {endpoint.method} {endpoint.path}\n")
                lines.append(f"**{endpoint.name}**\n")
                lines.append(endpoint.description)
                lines.append("")

                if endpoint.parameters:
                    lines.append("**Parameters:**\n")
                    lines.append("| Name | Type | Required | Description |")
                    lines.append("|------|------|----------|-------------|")
                    for param in endpoint.parameters:
                        req = "Yes" if param.required else "No"
                        lines.append(f"| {param.name} | {param.type} | {req} | {param.description} |")
                    lines.append("")

                if endpoint.examples:
                    lines.append("**Examples:**\n")
                    for ex in endpoint.examples:
                        lines.append("```json")
                        lines.append(json.dumps(ex, indent=2))
                        lines.append("```")
                    lines.append("")

        return "\n".join(lines)

    def to_json(self, doc: Document) -> str:
        """Convert document to JSON."""
        return json.dumps(doc.to_dict(), indent=2)

    def to_html(self, doc: Document) -> str:
        """Convert document to HTML."""
        markdown = self.to_markdown(doc)

        # Basic markdown to HTML conversion
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{doc.title}</title>",
            "<style>",
            "body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "pre { background: #f4f4f4; padding: 10px; overflow-x: auto; }",
            "code { background: #f4f4f4; padding: 2px 4px; }",
            "</style>",
            "</head>",
            "<body>",
        ]

        # Simple markdown conversion
        for line in markdown.split("\n"):
            if line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("**") and line.endswith("**"):
                html_lines.append(f"<p><strong>{line[2:-2]}</strong></p>")
            elif line.startswith("```"):
                if line == "```":
                    html_lines.append("</pre>")
                else:
                    html_lines.append("<pre>")
            elif line.startswith("|"):
                # Table handling simplified
                html_lines.append(f"<p>{line}</p>")
            elif line:
                html_lines.append(f"<p>{line}</p>")

        html_lines.extend(["</body>", "</html>"])
        return "\n".join(html_lines)


# ============================================================================
# Guide Generator
# ============================================================================

class GuideGenerator:
    """
    Generates user guides and tutorials.

    Example:
        generator = GuideGenerator()

        guide = generator.create_guide(
            "Getting Started",
            sections=[
                ("Installation", "Run pip install review-agent"),
                ("Configuration", "Create a config file..."),
            ],
        )
    """

    def __init__(self):
        """Initialize guide generator."""
        pass

    def create_guide(
        self,
        title: str,
        sections: List[tuple],
        doc_type: DocType = DocType.GUIDE,
    ) -> Document:
        """
        Create a guide document.

        Args:
            title: Guide title
            sections: List of (title, content) tuples
            doc_type: Type of documentation

        Returns:
            Document
        """
        doc = Document(
            title=title,
            doc_type=doc_type,
        )

        for section_title, section_content in sections:
            doc.sections.append(Section(
                title=section_title,
                content=section_content,
                level=2,
            ))

        return doc

    def to_markdown(self, doc: Document) -> str:
        """Convert guide to Markdown."""
        lines = [f"# {doc.title}\n"]

        for section in doc.sections:
            lines.append(f"{'#' * section.level} {section.title}\n")
            lines.append(section.content)
            lines.append("")

            for subsection in section.subsections:
                lines.append(f"{'#' * subsection.level} {subsection.title}\n")
                lines.append(subsection.content)
                lines.append("")

        return "\n".join(lines)


# ============================================================================
# Documentation Generator
# ============================================================================

class DocumentationGenerator:
    """
    Main documentation generation system.

    Example:
        generator = DocumentationGenerator()

        # Generate API docs
        api_doc = generator.generate_api_docs()

        # Generate user guide
        guide = generator.generate_guide()

        # Export
        generator.export(api_doc, DocFormat.MARKDOWN, "/path/to/docs")
    """

    BUS_TOPICS = {
        "generate": "review.docs.generate",
        "publish": "review.docs.publish",
    }

    def __init__(
        self,
        bus_path: Optional[Path] = None,
    ):
        """Initialize documentation generator."""
        self.bus_path = bus_path or self._get_bus_path()

        self.api_builder = APIDocBuilder()
        self.guide_generator = GuideGenerator()

        self._documents: Dict[str, Document] = {}
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "docs") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "documentation-generator",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def generate_api_docs(
        self,
        title: str = "Review Agent API",
        version: str = "1.0.0",
        description: str = "",
    ) -> Document:
        """
        Generate API documentation.

        Args:
            title: API title
            version: API version
            description: API description

        Returns:
            Generated Document
        """
        self.api_builder.set_info(title, version, description)

        # Add standard Review Agent endpoints
        self.api_builder.add_endpoint(Endpoint(
            name="Submit Review",
            method="POST",
            path="/api/v1/review",
            description="Submit files for code review",
            parameters=[
                Parameter("files", "array[string]", "File paths to review", required=True),
                Parameter("options", "object", "Review options"),
            ],
            tags=["review"],
        ))

        self.api_builder.add_endpoint(Endpoint(
            name="Get Review Status",
            method="GET",
            path="/api/v1/review/{review_id}",
            description="Get status of a review",
            parameters=[
                Parameter("review_id", "string", "Review identifier", required=True),
            ],
            tags=["review"],
        ))

        self.api_builder.add_endpoint(Endpoint(
            name="Health Check",
            method="GET",
            path="/api/v1/health",
            description="Check service health",
            tags=["system"],
        ))

        doc = self.api_builder.build()
        self._documents["api"] = doc

        self._emit_event(self.BUS_TOPICS["generate"], {
            "doc_type": "api",
            "title": title,
            "endpoints": len(doc.endpoints),
        })

        return doc

    def generate_guide(
        self,
        title: str = "Review Agent User Guide",
    ) -> Document:
        """
        Generate user guide.

        Args:
            title: Guide title

        Returns:
            Generated Document
        """
        sections = [
            ("Introduction", "The Review Agent provides automated code review capabilities."),
            ("Installation", "Install via pip:\n\n```bash\npip install review-agent\n```"),
            ("Configuration", "Configure via environment variables or config file."),
            ("Usage", "Submit files for review using the CLI or API."),
            ("Troubleshooting", "Common issues and solutions."),
        ]

        doc = self.guide_generator.create_guide(title, sections)
        self._documents["guide"] = doc

        self._emit_event(self.BUS_TOPICS["generate"], {
            "doc_type": "guide",
            "title": title,
            "sections": len(doc.sections),
        })

        return doc

    def export(
        self,
        doc: Document,
        format: DocFormat,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export document to file.

        Args:
            doc: Document to export
            format: Output format
            output_path: Output file path

        Returns:
            Generated content
        """
        if format == DocFormat.MARKDOWN:
            if doc.doc_type == DocType.API:
                content = self.api_builder.to_markdown(doc)
            else:
                content = self.guide_generator.to_markdown(doc)
            ext = ".md"
        elif format == DocFormat.HTML:
            content = self.api_builder.to_html(doc)
            ext = ".html"
        elif format == DocFormat.JSON:
            content = self.api_builder.to_json(doc)
            ext = ".json"
        else:
            content = self.api_builder.to_markdown(doc)
            ext = ".md"

        if output_path:
            path = Path(output_path)
            if path.suffix == "":
                path = path.with_suffix(ext)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

            self._emit_event(self.BUS_TOPICS["publish"], {
                "doc_type": doc.doc_type.value,
                "format": format.value,
                "path": str(path),
            })

        return content

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": "documentation-generator",
            "healthy": True,
            "documents_generated": len(self._documents),
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Documentation Generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Generator (Step 194)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # API command
    api_parser = subparsers.add_parser("api", help="Generate API docs")
    api_parser.add_argument("--title", default="Review Agent API", help="API title")
    api_parser.add_argument("--output", "-o", help="Output file path")
    api_parser.add_argument("--format", choices=["markdown", "html", "json"],
                            default="markdown", help="Output format")

    # Guide command
    guide_parser = subparsers.add_parser("guide", help="Generate user guide")
    guide_parser.add_argument("--title", default="Review Agent User Guide", help="Guide title")
    guide_parser.add_argument("--output", "-o", help="Output file path")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    generator = DocumentationGenerator()

    if args.command == "api":
        doc = generator.generate_api_docs(title=args.title)
        format = DocFormat[args.format.upper()]
        content = generator.export(doc, format, args.output)

        if not args.output:
            print(content)
        else:
            print(f"API documentation written to: {args.output}")

    elif args.command == "guide":
        doc = generator.generate_guide(title=args.title)
        content = generator.export(doc, DocFormat.MARKDOWN, args.output)

        if not args.output:
            print(content)
        else:
            print(f"User guide written to: {args.output}")

    else:
        # Default: show status
        status = generator.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Documentation Generator: {status['documents_generated']} documents")

    return 0


if __name__ == "__main__":
    sys.exit(main())
