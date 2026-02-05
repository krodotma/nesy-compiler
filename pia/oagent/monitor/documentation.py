#!/usr/bin/env python3
"""
Monitor Documentation Module - Step 294

API documentation and guides for the Monitor Agent.

PBTSO Phase: REPORT

Bus Topics:
- monitor.docs.generated (emitted)
- monitor.docs.exported (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import fcntl
import inspect
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type


class DocType(Enum):
    """Documentation types."""
    API = "api"
    GUIDE = "guide"
    REFERENCE = "reference"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"


class DocFormat(Enum):
    """Documentation output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    OPENAPI = "openapi"


@dataclass
class Parameter:
    """API parameter documentation.

    Attributes:
        name: Parameter name
        param_type: Parameter type
        description: Description
        required: Whether required
        default: Default value
        example: Example value
    """
    name: str
    param_type: str
    description: str = ""
    required: bool = True
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
class Response:
    """API response documentation.

    Attributes:
        status_code: HTTP status code
        description: Description
        schema: Response schema
        example: Example response
    """
    status_code: int
    description: str
    schema: Optional[Dict[str, Any]] = None
    example: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status_code": self.status_code,
            "description": self.description,
            "schema": self.schema,
            "example": self.example,
        }


@dataclass
class Endpoint:
    """API endpoint documentation.

    Attributes:
        path: Endpoint path
        method: HTTP method
        summary: Short summary
        description: Full description
        parameters: Request parameters
        request_body: Request body schema
        responses: Response definitions
        tags: Endpoint tags
        deprecated: Whether deprecated
        auth_required: Whether auth is required
    """
    path: str
    method: str
    summary: str
    description: str = ""
    parameters: List[Parameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: List[Response] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    auth_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "summary": self.summary,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "request_body": self.request_body,
            "responses": [r.to_dict() for r in self.responses],
            "tags": self.tags,
            "deprecated": self.deprecated,
            "auth_required": self.auth_required,
        }


@dataclass
class Section:
    """Documentation section.

    Attributes:
        title: Section title
        content: Section content
        subsections: Nested sections
        code_examples: Code examples
    """
    title: str
    content: str
    subsections: List["Section"] = field(default_factory=list)
    code_examples: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
            "code_examples": self.code_examples,
        }


@dataclass
class Document:
    """A documentation document.

    Attributes:
        title: Document title
        doc_type: Document type
        version: Document version
        description: Document description
        sections: Document sections
        endpoints: API endpoints
        created_at: Creation timestamp
        updated_at: Update timestamp
    """
    title: str
    doc_type: DocType
    version: str = "1.0.0"
    description: str = ""
    sections: List[Section] = field(default_factory=list)
    endpoints: List[Endpoint] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "type": self.doc_type.value,
            "version": self.version,
            "description": self.description,
            "sections": [s.to_dict() for s in self.sections],
            "endpoints": [e.to_dict() for e in self.endpoints],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MonitorDocumentation:
    """
    Documentation module for the Monitor Agent.

    Provides:
    - API documentation generation
    - OpenAPI spec generation
    - Markdown documentation
    - Code example extraction
    - Documentation export

    Example:
        docs = MonitorDocumentation()

        # Generate API docs
        api_doc = docs.generate_api_docs()

        # Export as markdown
        markdown = docs.export(api_doc, DocFormat.MARKDOWN)

        # Generate OpenAPI spec
        openapi = docs.generate_openapi()
    """

    BUS_TOPICS = {
        "generated": "monitor.docs.generated",
        "exported": "monitor.docs.exported",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        api_version: str = "v1",
        base_url: str = "http://localhost:8080",
        bus_dir: Optional[str] = None,
    ):
        """Initialize documentation module.

        Args:
            api_version: API version
            base_url: Base URL for API
            bus_dir: Bus directory
        """
        self._api_version = api_version
        self._base_url = base_url
        self._last_heartbeat = time.time()

        # Document registry
        self._documents: Dict[str, Document] = {}
        self._endpoints: List[Endpoint] = []

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default endpoints
        self._register_default_endpoints()

    def register_endpoint(self, endpoint: Endpoint) -> None:
        """Register an API endpoint.

        Args:
            endpoint: Endpoint to register
        """
        self._endpoints.append(endpoint)

    def document_class(
        self,
        cls: Type,
        include_private: bool = False,
    ) -> Document:
        """Generate documentation for a class.

        Args:
            cls: Class to document
            include_private: Include private methods

        Returns:
            Generated document
        """
        sections = []

        # Class docstring
        class_doc = inspect.getdoc(cls) or ""
        sections.append(Section(
            title="Overview",
            content=class_doc,
        ))

        # Methods
        method_sections = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_") and not include_private:
                continue

            method_doc = inspect.getdoc(method) or ""
            sig = inspect.signature(method)

            params = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                param_type = (
                    param.annotation.__name__
                    if param.annotation != inspect.Parameter.empty
                    else "Any"
                )
                params.append(f"- `{param_name}`: {param_type}")

            method_sections.append(Section(
                title=f"`{name}()`",
                content=method_doc + "\n\n" + "\n".join(params) if params else method_doc,
            ))

        if method_sections:
            sections.append(Section(
                title="Methods",
                content="",
                subsections=method_sections,
            ))

        return Document(
            title=cls.__name__,
            doc_type=DocType.REFERENCE,
            description=class_doc.split("\n")[0] if class_doc else "",
            sections=sections,
        )

    def generate_api_docs(self) -> Document:
        """Generate API documentation.

        Returns:
            API documentation document
        """
        doc = Document(
            title="Monitor Agent API",
            doc_type=DocType.API,
            version=self._api_version,
            description="REST API for the Monitor Agent",
            endpoints=self._endpoints,
        )

        # Group endpoints by tag
        tags: Dict[str, List[Endpoint]] = {}
        for endpoint in self._endpoints:
            for tag in endpoint.tags or ["General"]:
                if tag not in tags:
                    tags[tag] = []
                tags[tag].append(endpoint)

        # Create sections
        for tag, endpoints in tags.items():
            section_content = []
            for ep in endpoints:
                section_content.append(f"### {ep.method} {ep.path}\n")
                section_content.append(f"{ep.summary}\n")
                if ep.description:
                    section_content.append(f"\n{ep.description}\n")
                if ep.auth_required:
                    section_content.append("\n**Authentication required**\n")
                if ep.deprecated:
                    section_content.append("\n**DEPRECATED**\n")

            doc.sections.append(Section(
                title=tag,
                content="\n".join(section_content),
            ))

        self._emit_bus_event(
            self.BUS_TOPICS["generated"],
            {"doc_type": "api", "endpoints": len(self._endpoints)},
        )

        return doc

    def generate_openapi(self) -> Dict[str, Any]:
        """Generate OpenAPI specification.

        Returns:
            OpenAPI spec dictionary
        """
        paths: Dict[str, Dict[str, Any]] = {}

        for endpoint in self._endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}

            method = endpoint.method.lower()
            paths[endpoint.path][method] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "deprecated": endpoint.deprecated,
                "parameters": [
                    {
                        "name": p.name,
                        "in": "query",
                        "required": p.required,
                        "schema": {"type": p.param_type},
                        "description": p.description,
                    }
                    for p in endpoint.parameters
                ],
                "responses": {
                    str(r.status_code): {
                        "description": r.description,
                    }
                    for r in endpoint.responses
                },
            }

            if endpoint.request_body:
                paths[endpoint.path][method]["requestBody"] = {
                    "content": {
                        "application/json": {
                            "schema": endpoint.request_body,
                        }
                    }
                }

            if endpoint.auth_required:
                paths[endpoint.path][method]["security"] = [{"apiKey": []}]

        openapi = {
            "openapi": "3.0.3",
            "info": {
                "title": "Monitor Agent API",
                "version": self._api_version,
                "description": "REST API for the Monitor Agent",
            },
            "servers": [
                {"url": self._base_url, "description": "Default server"}
            ],
            "paths": paths,
            "components": {
                "securitySchemes": {
                    "apiKey": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                    }
                }
            },
        }

        self._emit_bus_event(
            self.BUS_TOPICS["generated"],
            {"doc_type": "openapi", "paths": len(paths)},
        )

        return openapi

    def generate_guide(
        self,
        title: str,
        sections: List[Section],
    ) -> Document:
        """Generate a guide document.

        Args:
            title: Guide title
            sections: Guide sections

        Returns:
            Guide document
        """
        return Document(
            title=title,
            doc_type=DocType.GUIDE,
            sections=sections,
        )

    def export(
        self,
        document: Document,
        format: DocFormat = DocFormat.MARKDOWN,
    ) -> str:
        """Export document to specified format.

        Args:
            document: Document to export
            format: Output format

        Returns:
            Exported document string
        """
        if format == DocFormat.MARKDOWN:
            return self._export_markdown(document)
        elif format == DocFormat.HTML:
            return self._export_html(document)
        elif format == DocFormat.JSON:
            return json.dumps(document.to_dict(), indent=2)
        elif format == DocFormat.OPENAPI:
            return json.dumps(self.generate_openapi(), indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

    def save(
        self,
        document: Document,
        path: str,
        format: DocFormat = DocFormat.MARKDOWN,
    ) -> bool:
        """Save document to file.

        Args:
            document: Document to save
            path: Output path
            format: Output format

        Returns:
            True if saved
        """
        content = self.export(document, format)

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)

            self._emit_bus_event(
                self.BUS_TOPICS["exported"],
                {"path": path, "format": format.value},
            )

            return True
        except Exception:
            return False

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all registered documents.

        Returns:
            Document info list
        """
        return [
            {
                "title": doc.title,
                "type": doc.doc_type.value,
                "version": doc.version,
                "sections": len(doc.sections),
                "updated_at": doc.updated_at,
            }
            for doc in self._documents.values()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get documentation statistics.

        Returns:
            Statistics
        """
        return {
            "total_documents": len(self._documents),
            "total_endpoints": len(self._endpoints),
            "by_type": {
                t.value: sum(
                    1 for d in self._documents.values()
                    if d.doc_type == t
                )
                for t in DocType
            },
            "endpoints_by_method": {
                method: sum(1 for e in self._endpoints if e.method == method)
                for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]
            },
        }

    def _register_default_endpoints(self) -> None:
        """Register default API endpoints."""
        endpoints = [
            Endpoint(
                path="/api/v1/health",
                method="GET",
                summary="Health check",
                description="Check if the Monitor Agent is healthy",
                responses=[
                    Response(200, "Healthy"),
                    Response(503, "Unhealthy"),
                ],
                tags=["Health"],
            ),
            Endpoint(
                path="/api/v1/metrics",
                method="GET",
                summary="List metrics",
                description="Get a list of all available metrics",
                parameters=[
                    Parameter("prefix", "string", "Filter by prefix", required=False),
                    Parameter("limit", "integer", "Maximum results", required=False, default=100),
                ],
                responses=[Response(200, "List of metrics")],
                tags=["Metrics"],
            ),
            Endpoint(
                path="/api/v1/metrics/query",
                method="GET",
                summary="Query metric",
                description="Query a specific metric value",
                parameters=[
                    Parameter("name", "string", "Metric name", required=True),
                    Parameter("agg", "string", "Aggregation function", required=False, default="avg"),
                    Parameter("window", "integer", "Time window in seconds", required=False, default=300),
                ],
                responses=[Response(200, "Metric value")],
                tags=["Metrics"],
            ),
            Endpoint(
                path="/api/v1/metrics",
                method="POST",
                summary="Record metric",
                description="Record a new metric point",
                request_body={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "number"},
                        "labels": {"type": "object"},
                    },
                    "required": ["name", "value"],
                },
                responses=[
                    Response(201, "Metric recorded"),
                    Response(400, "Invalid request"),
                ],
                tags=["Metrics"],
                auth_required=True,
            ),
            Endpoint(
                path="/api/v1/alerts",
                method="GET",
                summary="List alerts",
                description="Get a list of alerts",
                parameters=[
                    Parameter("state", "string", "Filter by state", required=False, default="firing"),
                    Parameter("severity", "string", "Filter by severity", required=False),
                ],
                responses=[Response(200, "List of alerts")],
                tags=["Alerts"],
            ),
            Endpoint(
                path="/api/v1/alerts/{id}",
                method="GET",
                summary="Get alert",
                description="Get alert by ID",
                parameters=[Parameter("id", "string", "Alert ID", required=True)],
                responses=[
                    Response(200, "Alert details"),
                    Response(404, "Alert not found"),
                ],
                tags=["Alerts"],
            ),
            Endpoint(
                path="/api/v1/alerts/{id}/acknowledge",
                method="POST",
                summary="Acknowledge alert",
                description="Acknowledge an alert",
                parameters=[Parameter("id", "string", "Alert ID", required=True)],
                responses=[Response(200, "Alert acknowledged")],
                tags=["Alerts"],
                auth_required=True,
            ),
            Endpoint(
                path="/api/v1/dashboards",
                method="GET",
                summary="List dashboards",
                description="Get a list of dashboards",
                responses=[Response(200, "List of dashboards")],
                tags=["Dashboards"],
            ),
            Endpoint(
                path="/api/v1/reports",
                method="POST",
                summary="Generate report",
                description="Generate a new report",
                request_body={
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "period": {"type": "string"},
                    },
                },
                responses=[Response(201, "Report generation started")],
                tags=["Reports"],
                auth_required=True,
            ),
        ]

        for ep in endpoints:
            self._endpoints.append(ep)

    def _export_markdown(self, document: Document) -> str:
        """Export document as Markdown."""
        lines = [
            f"# {document.title}",
            "",
            f"*Version: {document.version}*",
            "",
            document.description,
            "",
        ]

        # Add sections
        for section in document.sections:
            lines.extend(self._section_to_markdown(section, level=2))

        # Add endpoints
        if document.endpoints:
            lines.append("## API Endpoints")
            lines.append("")

            for endpoint in document.endpoints:
                lines.append(f"### {endpoint.method} `{endpoint.path}`")
                lines.append("")
                lines.append(f"**{endpoint.summary}**")
                lines.append("")

                if endpoint.description:
                    lines.append(endpoint.description)
                    lines.append("")

                if endpoint.deprecated:
                    lines.append("> **DEPRECATED**: This endpoint is deprecated.")
                    lines.append("")

                if endpoint.auth_required:
                    lines.append("*Requires authentication*")
                    lines.append("")

                if endpoint.parameters:
                    lines.append("**Parameters:**")
                    lines.append("")
                    for p in endpoint.parameters:
                        req = "required" if p.required else "optional"
                        lines.append(f"- `{p.name}` ({p.param_type}, {req}): {p.description}")
                    lines.append("")

                if endpoint.responses:
                    lines.append("**Responses:**")
                    lines.append("")
                    for r in endpoint.responses:
                        lines.append(f"- `{r.status_code}`: {r.description}")
                    lines.append("")

        return "\n".join(lines)

    def _section_to_markdown(
        self,
        section: Section,
        level: int = 2,
    ) -> List[str]:
        """Convert section to Markdown lines."""
        lines = [
            "#" * level + " " + section.title,
            "",
        ]

        if section.content:
            lines.append(section.content)
            lines.append("")

        for example in section.code_examples:
            lang = example.get("language", "")
            code = example.get("code", "")
            lines.append(f"```{lang}")
            lines.append(code)
            lines.append("```")
            lines.append("")

        for subsection in section.subsections:
            lines.extend(self._section_to_markdown(subsection, level + 1))

        return lines

    def _export_html(self, document: Document) -> str:
        """Export document as HTML."""
        markdown = self._export_markdown(document)

        # Simple HTML conversion
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{document.title}</title>",
            "<style>",
            "body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }",
            "pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }",
            "</style>",
            "</head>",
            "<body>",
        ]

        # Simple markdown to HTML (basic conversion)
        for line in markdown.split("\n"):
            if line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("- "):
                html_lines.append(f"<li>{line[2:]}</li>")
            elif line.startswith("```"):
                html_lines.append("<pre><code>")
            elif line == "```":
                html_lines.append("</code></pre>")
            elif line:
                html_lines.append(f"<p>{line}</p>")

        html_lines.extend([
            "</body>",
            "</html>",
        ])

        return "\n".join(html_lines)

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_documentation",
                "status": "healthy",
                "endpoints": len(self._endpoints),
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-documentation",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_documentation: Optional[MonitorDocumentation] = None


def get_documentation() -> MonitorDocumentation:
    """Get or create the documentation module singleton.

    Returns:
        MonitorDocumentation instance
    """
    global _documentation
    if _documentation is None:
        _documentation = MonitorDocumentation()
    return _documentation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Documentation Module (Step 294)")
    parser.add_argument("--generate", choices=["api", "openapi"], help="Generate documentation")
    parser.add_argument("--format", choices=["markdown", "html", "json"], default="markdown", help="Output format")
    parser.add_argument("--output", metavar="FILE", help="Output file")
    parser.add_argument("--list", action="store_true", help="List endpoints")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    docs = get_documentation()

    if args.generate == "api":
        api_doc = docs.generate_api_docs()
        format = DocFormat(args.format)
        output = docs.export(api_doc, format)

        if args.output:
            docs.save(api_doc, args.output, format)
            print(f"Saved to {args.output}")
        else:
            print(output)

    if args.generate == "openapi":
        openapi = docs.generate_openapi()
        output = json.dumps(openapi, indent=2)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Saved to {args.output}")
        else:
            print(output)

    if args.list:
        endpoints = docs._endpoints
        if args.json:
            print(json.dumps([e.to_dict() for e in endpoints], indent=2))
        else:
            print("API Endpoints:")
            for ep in endpoints:
                auth = " (auth)" if ep.auth_required else ""
                print(f"  {ep.method:6} {ep.path}{auth}")
                print(f"         {ep.summary}")

    if args.stats:
        stats = docs.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Documentation Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
