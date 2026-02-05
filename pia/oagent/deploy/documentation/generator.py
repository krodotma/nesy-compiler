#!/usr/bin/env python3
"""
generator.py - Documentation Generator (Step 244)

PBTSO Phase: OBSERVE
A2A Integration: Generates documentation via deploy.docs.generate

Provides:
- DocFormat: Documentation formats
- DocSection: Documentation section
- APIDoc: API documentation
- Guide: User guide
- DocIndex: Documentation index
- DocumentationGenerator: Complete documentation generation

Bus Topics:
- deploy.docs.generate
- deploy.docs.publish
- deploy.docs.index

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ==============================================================================
# Bus Emission Helper with fcntl.flock()
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "documentation-generator"
) -> str:
    """Emit an event to the Pluribus bus with file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class DocFormat(Enum):
    """Documentation output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    OPENAPI = "openapi"
    ASYNCAPI = "asyncapi"


class DocType(Enum):
    """Types of documentation."""
    API = "api"
    GUIDE = "guide"
    REFERENCE = "reference"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"
    ARCHITECTURE = "architecture"


@dataclass
class DocSection:
    """
    Documentation section.

    Attributes:
        section_id: Unique section identifier
        title: Section title
        content: Section content
        order: Sort order
        subsections: Child sections
        metadata: Additional metadata
    """
    section_id: str
    title: str
    content: str = ""
    order: int = 0
    subsections: List["DocSection"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "order": self.order,
            "subsections": [s.to_dict() for s in self.subsections],
            "metadata": self.metadata,
        }


@dataclass
class APIEndpoint:
    """
    API endpoint documentation.

    Attributes:
        path: Endpoint path
        method: HTTP method
        summary: Short summary
        description: Full description
        parameters: Request parameters
        request_body: Request body schema
        responses: Response schemas
        tags: Endpoint tags
        deprecated: Whether deprecated
        examples: Usage examples
    """
    path: str
    method: str = "GET"
    summary: str = ""
    description: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class APIDoc:
    """
    API documentation.

    Attributes:
        doc_id: Unique document identifier
        title: API title
        version: API version
        description: API description
        base_url: Base URL
        endpoints: API endpoints
        schemas: Data schemas
        auth_methods: Authentication methods
        created_at: Creation timestamp
        updated_at: Update timestamp
    """
    doc_id: str
    title: str
    version: str = "1.0.0"
    description: str = ""
    base_url: str = ""
    endpoints: List[APIEndpoint] = field(default_factory=list)
    schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    auth_methods: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "version": self.version,
            "description": self.description,
            "base_url": self.base_url,
            "endpoints": [e.to_dict() for e in self.endpoints],
            "schemas": self.schemas,
            "auth_methods": self.auth_methods,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_openapi(self) -> Dict[str, Any]:
        """Convert to OpenAPI 3.0 format."""
        paths = {}
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}

            operation = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "deprecated": endpoint.deprecated,
                "parameters": endpoint.parameters,
                "responses": endpoint.responses or {"200": {"description": "Success"}},
            }

            if endpoint.request_body:
                operation["requestBody"] = endpoint.request_body

            paths[endpoint.path][endpoint.method.lower()] = operation

        return {
            "openapi": "3.0.3",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": self.description,
            },
            "servers": [{"url": self.base_url}] if self.base_url else [],
            "paths": paths,
            "components": {
                "schemas": self.schemas,
                "securitySchemes": {
                    method.get("name", "default"): method
                    for method in self.auth_methods
                } if self.auth_methods else {},
            },
        }


@dataclass
class Guide:
    """
    User guide documentation.

    Attributes:
        guide_id: Unique guide identifier
        title: Guide title
        description: Guide description
        doc_type: Type of guide
        sections: Guide sections
        prerequisites: Prerequisites
        difficulty: Difficulty level
        estimated_time: Estimated completion time
        created_at: Creation timestamp
        updated_at: Update timestamp
    """
    guide_id: str
    title: str
    description: str = ""
    doc_type: DocType = DocType.GUIDE
    sections: List[DocSection] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    difficulty: str = "beginner"
    estimated_time: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guide_id": self.guide_id,
            "title": self.title,
            "description": self.description,
            "doc_type": self.doc_type.value,
            "sections": [s.to_dict() for s in self.sections],
            "prerequisites": self.prerequisites,
            "difficulty": self.difficulty,
            "estimated_time": self.estimated_time,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class DocIndex:
    """
    Documentation index/table of contents.

    Attributes:
        index_id: Unique index identifier
        title: Index title
        entries: Index entries
        created_at: Creation timestamp
    """
    index_id: str
    title: str = "Documentation Index"
    entries: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def add_entry(
        self,
        title: str,
        path: str,
        doc_type: DocType,
        description: str = "",
    ) -> None:
        """Add an entry to the index."""
        self.entries.append({
            "title": title,
            "path": path,
            "type": doc_type.value,
            "description": description,
        })


# ==============================================================================
# Documentation Generator (Step 244)
# ==============================================================================

class DocumentationGenerator:
    """
    Documentation Generator - API docs and guides for deployments.

    PBTSO Phase: OBSERVE

    Responsibilities:
    - Generate API documentation
    - Create user guides
    - Build documentation indexes
    - Export to multiple formats

    Example:
        >>> generator = DocumentationGenerator()
        >>> api_doc = generator.create_api_doc(
        ...     title="Deploy API",
        ...     version="v1.0.0",
        ...     base_url="https://api.example.com"
        ... )
        >>> generator.add_endpoint(api_doc.doc_id, APIEndpoint(
        ...     path="/deploy",
        ...     method="POST",
        ...     summary="Deploy a service"
        ... ))
        >>> openapi = generator.export(api_doc.doc_id, DocFormat.OPENAPI)
    """

    BUS_TOPICS = {
        "generate": "deploy.docs.generate",
        "publish": "deploy.docs.publish",
        "index": "deploy.docs.index",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        actor_id: str = "documentation-generator",
    ):
        """
        Initialize the documentation generator.

        Args:
            state_dir: Directory for state persistence
            output_dir: Directory for generated docs
            actor_id: Actor identifier for bus events
        """
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))

        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "docs"

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = pluribus_root / ".pluribus" / "deploy" / "docs" / "output"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Storage
        self._api_docs: Dict[str, APIDoc] = {}
        self._guides: Dict[str, Guide] = {}
        self._index: Optional[DocIndex] = None

        self._load_state()

    def create_api_doc(
        self,
        title: str,
        version: str = "1.0.0",
        description: str = "",
        base_url: str = "",
    ) -> APIDoc:
        """
        Create a new API documentation.

        Args:
            title: API title
            version: API version
            description: API description
            base_url: Base URL

        Returns:
            Created APIDoc
        """
        doc_id = f"api-{uuid.uuid4().hex[:12]}"

        doc = APIDoc(
            doc_id=doc_id,
            title=title,
            version=version,
            description=description,
            base_url=base_url,
        )

        self._api_docs[doc_id] = doc
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["generate"],
            {
                "action": "api_doc_created",
                "doc_id": doc_id,
                "title": title,
                "version": version,
            },
            actor=self.actor_id,
        )

        return doc

    def add_endpoint(self, doc_id: str, endpoint: APIEndpoint) -> bool:
        """
        Add an endpoint to an API doc.

        Args:
            doc_id: API doc ID
            endpoint: Endpoint to add

        Returns:
            True if added successfully
        """
        doc = self._api_docs.get(doc_id)
        if not doc:
            return False

        doc.endpoints.append(endpoint)
        doc.updated_at = time.time()
        self._save_state()

        return True

    def add_schema(
        self,
        doc_id: str,
        name: str,
        schema: Dict[str, Any],
    ) -> bool:
        """
        Add a schema to an API doc.

        Args:
            doc_id: API doc ID
            name: Schema name
            schema: JSON schema

        Returns:
            True if added successfully
        """
        doc = self._api_docs.get(doc_id)
        if not doc:
            return False

        doc.schemas[name] = schema
        doc.updated_at = time.time()
        self._save_state()

        return True

    def create_guide(
        self,
        title: str,
        description: str = "",
        doc_type: DocType = DocType.GUIDE,
        difficulty: str = "beginner",
        estimated_time: str = "",
    ) -> Guide:
        """
        Create a new guide.

        Args:
            title: Guide title
            description: Guide description
            doc_type: Type of guide
            difficulty: Difficulty level
            estimated_time: Estimated completion time

        Returns:
            Created Guide
        """
        guide_id = f"guide-{uuid.uuid4().hex[:12]}"

        guide = Guide(
            guide_id=guide_id,
            title=title,
            description=description,
            doc_type=doc_type,
            difficulty=difficulty,
            estimated_time=estimated_time,
        )

        self._guides[guide_id] = guide
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["generate"],
            {
                "action": "guide_created",
                "guide_id": guide_id,
                "title": title,
                "type": doc_type.value,
            },
            actor=self.actor_id,
        )

        return guide

    def add_section(
        self,
        guide_id: str,
        title: str,
        content: str,
        order: int = 0,
        parent_section_id: Optional[str] = None,
    ) -> Optional[DocSection]:
        """
        Add a section to a guide.

        Args:
            guide_id: Guide ID
            title: Section title
            content: Section content
            order: Sort order
            parent_section_id: Parent section (for nesting)

        Returns:
            Created DocSection or None
        """
        guide = self._guides.get(guide_id)
        if not guide:
            return None

        section_id = f"section-{uuid.uuid4().hex[:8]}"
        section = DocSection(
            section_id=section_id,
            title=title,
            content=content,
            order=order,
        )

        if parent_section_id:
            parent = self._find_section(guide.sections, parent_section_id)
            if parent:
                parent.subsections.append(section)
        else:
            guide.sections.append(section)

        guide.updated_at = time.time()
        self._save_state()

        return section

    def _find_section(
        self,
        sections: List[DocSection],
        section_id: str,
    ) -> Optional[DocSection]:
        """Find a section by ID."""
        for section in sections:
            if section.section_id == section_id:
                return section
            found = self._find_section(section.subsections, section_id)
            if found:
                return found
        return None

    def export(
        self,
        doc_id: str,
        format: DocFormat = DocFormat.MARKDOWN,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export documentation to a specific format.

        Args:
            doc_id: Document ID (API doc or guide)
            format: Output format
            output_path: Output file path

        Returns:
            Exported content string
        """
        content = ""

        # Check if it's an API doc
        if doc_id in self._api_docs:
            doc = self._api_docs[doc_id]

            if format == DocFormat.OPENAPI:
                content = json.dumps(doc.to_openapi(), indent=2)
            elif format == DocFormat.JSON:
                content = json.dumps(doc.to_dict(), indent=2)
            elif format == DocFormat.HTML:
                content = self._api_to_html(doc)
            else:  # MARKDOWN
                content = self._api_to_markdown(doc)

        # Check if it's a guide
        elif doc_id in self._guides:
            guide = self._guides[doc_id]

            if format == DocFormat.JSON:
                content = json.dumps(guide.to_dict(), indent=2)
            elif format == DocFormat.HTML:
                content = self._guide_to_html(guide)
            else:  # MARKDOWN
                content = self._guide_to_markdown(guide)

        else:
            raise ValueError(f"Document not found: {doc_id}")

        # Write to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content)

            _emit_bus_event(
                self.BUS_TOPICS["publish"],
                {
                    "doc_id": doc_id,
                    "format": format.value,
                    "path": output_path,
                },
                actor=self.actor_id,
            )

        return content

    def _api_to_markdown(self, doc: APIDoc) -> str:
        """Convert API doc to Markdown."""
        lines = [
            f"# {doc.title}",
            "",
            f"**Version:** {doc.version}",
            "",
            doc.description,
            "",
        ]

        if doc.base_url:
            lines.extend([f"**Base URL:** `{doc.base_url}`", ""])

        if doc.auth_methods:
            lines.extend(["## Authentication", ""])
            for method in doc.auth_methods:
                lines.append(f"- **{method.get('name', 'Auth')}**: {method.get('description', '')}")
            lines.append("")

        lines.extend(["## Endpoints", ""])

        for endpoint in doc.endpoints:
            deprecated = " (DEPRECATED)" if endpoint.deprecated else ""
            lines.extend([
                f"### {endpoint.method} {endpoint.path}{deprecated}",
                "",
                endpoint.summary,
                "",
            ])

            if endpoint.description:
                lines.extend([endpoint.description, ""])

            if endpoint.parameters:
                lines.extend(["**Parameters:**", ""])
                for param in endpoint.parameters:
                    required = " (required)" if param.get("required") else ""
                    lines.append(f"- `{param.get('name')}`: {param.get('description', '')}{required}")
                lines.append("")

            if endpoint.responses:
                lines.extend(["**Responses:**", ""])
                for code, response in endpoint.responses.items():
                    lines.append(f"- `{code}`: {response.get('description', '')}")
                lines.append("")

        if doc.schemas:
            lines.extend(["## Schemas", ""])
            for name, schema in doc.schemas.items():
                lines.extend([
                    f"### {name}",
                    "",
                    "```json",
                    json.dumps(schema, indent=2),
                    "```",
                    "",
                ])

        return "\n".join(lines)

    def _api_to_html(self, doc: APIDoc) -> str:
        """Convert API doc to HTML."""
        md_content = self._api_to_markdown(doc)
        # Simple HTML conversion
        html = self._markdown_to_html(md_content)
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{doc.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        h3 {{ color: #666; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .deprecated {{ color: #d32f2f; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""

    def _guide_to_markdown(self, guide: Guide) -> str:
        """Convert guide to Markdown."""
        lines = [
            f"# {guide.title}",
            "",
            guide.description,
            "",
        ]

        if guide.prerequisites:
            lines.extend(["## Prerequisites", ""])
            for prereq in guide.prerequisites:
                lines.append(f"- {prereq}")
            lines.append("")

        if guide.difficulty or guide.estimated_time:
            lines.extend([
                f"**Difficulty:** {guide.difficulty}",
                f"**Estimated Time:** {guide.estimated_time}",
                "",
            ])

        def render_sections(sections: List[DocSection], level: int = 2) -> None:
            for section in sorted(sections, key=lambda s: s.order):
                prefix = "#" * level
                lines.extend([
                    f"{prefix} {section.title}",
                    "",
                    section.content,
                    "",
                ])
                render_sections(section.subsections, level + 1)

        render_sections(guide.sections)

        return "\n".join(lines)

    def _guide_to_html(self, guide: Guide) -> str:
        """Convert guide to HTML."""
        md_content = self._guide_to_markdown(guide)
        html = self._markdown_to_html(md_content)
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{guide.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; max-width: 800px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
{html}
</body>
</html>"""

    def _markdown_to_html(self, md: str) -> str:
        """Simple Markdown to HTML conversion."""
        html = md

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Code blocks
        html = re.sub(r'```(\w+)?\n(.*?)```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)

        # Inline code
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

        # Lists
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Paragraphs
        paragraphs = html.split('\n\n')
        html = '\n'.join(f'<p>{p}</p>' if not p.startswith('<') else p for p in paragraphs)

        return html

    def build_index(self) -> DocIndex:
        """
        Build documentation index.

        Returns:
            DocIndex
        """
        index = DocIndex(
            index_id=f"index-{uuid.uuid4().hex[:8]}",
            title="Documentation Index",
        )

        # Add API docs
        for doc in self._api_docs.values():
            index.add_entry(
                title=doc.title,
                path=f"api/{doc.doc_id}",
                doc_type=DocType.API,
                description=doc.description[:100],
            )

        # Add guides
        for guide in self._guides.values():
            index.add_entry(
                title=guide.title,
                path=f"guides/{guide.guide_id}",
                doc_type=guide.doc_type,
                description=guide.description[:100],
            )

        self._index = index

        _emit_bus_event(
            self.BUS_TOPICS["index"],
            {
                "index_id": index.index_id,
                "entry_count": len(index.entries),
            },
            actor=self.actor_id,
        )

        return index

    def publish_all(
        self,
        format: DocFormat = DocFormat.MARKDOWN,
    ) -> Dict[str, str]:
        """
        Publish all documentation.

        Args:
            format: Output format

        Returns:
            Dict of doc_id -> output_path
        """
        outputs = {}
        ext = {
            DocFormat.MARKDOWN: ".md",
            DocFormat.HTML: ".html",
            DocFormat.JSON: ".json",
            DocFormat.OPENAPI: ".json",
        }.get(format, ".md")

        for doc_id in self._api_docs:
            output_path = str(self.output_dir / f"api/{doc_id}{ext}")
            self.export(doc_id, format, output_path)
            outputs[doc_id] = output_path

        for guide_id in self._guides:
            output_path = str(self.output_dir / f"guides/{guide_id}{ext}")
            self.export(guide_id, format, output_path)
            outputs[guide_id] = output_path

        # Build and save index
        index = self.build_index()
        index_path = str(self.output_dir / "index.json")
        Path(index_path).write_text(json.dumps(index.to_dict(), indent=2))
        outputs["index"] = index_path

        return outputs

    def get_api_doc(self, doc_id: str) -> Optional[APIDoc]:
        """Get an API doc by ID."""
        return self._api_docs.get(doc_id)

    def get_guide(self, guide_id: str) -> Optional[Guide]:
        """Get a guide by ID."""
        return self._guides.get(guide_id)

    def list_api_docs(self) -> List[APIDoc]:
        """List all API docs."""
        return list(self._api_docs.values())

    def list_guides(self) -> List[Guide]:
        """List all guides."""
        return list(self._guides.values())

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "api_docs": {k: v.to_dict() for k, v in self._api_docs.items()},
            "guides": {k: v.to_dict() for k, v in self._guides.items()},
        }
        state_file = self.state_dir / "docs_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "docs_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for k, v in state.get("api_docs", {}).items():
                doc = APIDoc(
                    doc_id=v["doc_id"],
                    title=v["title"],
                    version=v.get("version", "1.0.0"),
                    description=v.get("description", ""),
                    base_url=v.get("base_url", ""),
                )
                for ep_data in v.get("endpoints", []):
                    doc.endpoints.append(APIEndpoint(**ep_data))
                doc.schemas = v.get("schemas", {})
                doc.auth_methods = v.get("auth_methods", [])
                self._api_docs[k] = doc

            for k, v in state.get("guides", {}).items():
                guide = Guide(
                    guide_id=v["guide_id"],
                    title=v["title"],
                    description=v.get("description", ""),
                    doc_type=DocType(v.get("doc_type", "guide")),
                    difficulty=v.get("difficulty", "beginner"),
                    estimated_time=v.get("estimated_time", ""),
                )
                self._guides[k] = guide

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for documentation generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Generator (Step 244)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-api command
    api_parser = subparsers.add_parser("create-api", help="Create API documentation")
    api_parser.add_argument("title", help="API title")
    api_parser.add_argument("--version", "-v", default="1.0.0", help="API version")
    api_parser.add_argument("--description", "-d", default="", help="Description")
    api_parser.add_argument("--base-url", "-u", default="", help="Base URL")
    api_parser.add_argument("--json", action="store_true", help="JSON output")

    # create-guide command
    guide_parser = subparsers.add_parser("create-guide", help="Create a guide")
    guide_parser.add_argument("title", help="Guide title")
    guide_parser.add_argument("--description", "-d", default="", help="Description")
    guide_parser.add_argument("--type", "-t", default="guide",
                             choices=["guide", "tutorial", "reference", "changelog"])
    guide_parser.add_argument("--json", action="store_true", help="JSON output")

    # export command
    export_parser = subparsers.add_parser("export", help="Export documentation")
    export_parser.add_argument("doc_id", help="Document ID")
    export_parser.add_argument("--format", "-f", default="markdown",
                              choices=["markdown", "html", "json", "openapi"])
    export_parser.add_argument("--output", "-o", help="Output file")

    # publish command
    publish_parser = subparsers.add_parser("publish", help="Publish all documentation")
    publish_parser.add_argument("--format", "-f", default="markdown",
                               choices=["markdown", "html", "json"])

    # list command
    list_parser = subparsers.add_parser("list", help="List documentation")
    list_parser.add_argument("--type", "-t", choices=["api", "guide", "all"], default="all")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    generator = DocumentationGenerator()

    if args.command == "create-api":
        doc = generator.create_api_doc(
            title=args.title,
            version=args.version,
            description=args.description,
            base_url=args.base_url,
        )

        if args.json:
            print(json.dumps(doc.to_dict(), indent=2))
        else:
            print(f"Created API doc: {doc.doc_id}")
            print(f"  Title: {doc.title}")
            print(f"  Version: {doc.version}")

        return 0

    elif args.command == "create-guide":
        guide = generator.create_guide(
            title=args.title,
            description=args.description,
            doc_type=DocType(args.type),
        )

        if args.json:
            print(json.dumps(guide.to_dict(), indent=2))
        else:
            print(f"Created guide: {guide.guide_id}")
            print(f"  Title: {guide.title}")
            print(f"  Type: {guide.doc_type.value}")

        return 0

    elif args.command == "export":
        try:
            content = generator.export(
                doc_id=args.doc_id,
                format=DocFormat(args.format),
                output_path=args.output,
            )

            if not args.output:
                print(content)
            else:
                print(f"Exported to {args.output}")

            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "publish":
        outputs = generator.publish_all(format=DocFormat(args.format))
        print(f"Published {len(outputs)} documents:")
        for doc_id, path in outputs.items():
            print(f"  {doc_id}: {path}")
        return 0

    elif args.command == "list":
        docs = []
        if args.type in ("api", "all"):
            for doc in generator.list_api_docs():
                docs.append({"id": doc.doc_id, "title": doc.title, "type": "api"})
        if args.type in ("guide", "all"):
            for guide in generator.list_guides():
                docs.append({"id": guide.guide_id, "title": guide.title, "type": guide.doc_type.value})

        if args.json:
            print(json.dumps(docs, indent=2))
        else:
            if not docs:
                print("No documentation found")
            else:
                for d in docs:
                    print(f"{d['id']} ({d['type']}): {d['title']}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
