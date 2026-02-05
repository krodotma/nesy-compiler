#!/usr/bin/env python3
"""
Review Template Manager (Step 175)

Manages review templates for consistent code review processes.

PBTSO Phase: SKILL, BUILD
Bus Topics: review.template.apply, review.template.create

Features:
- Predefined review templates
- Custom template creation
- Template variables
- Category-specific templates
- Team/project templates

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


# ============================================================================
# Types
# ============================================================================

class TemplateCategory(Enum):
    """Categories of review templates."""
    GENERAL = "general"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    API = "api"
    DATABASE = "database"
    FRONTEND = "frontend"
    BACKEND = "backend"
    TESTING = "testing"


class VariableType(Enum):
    """Types of template variables."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    DATE = "date"
    FILE = "file"


@dataclass
class TemplateVariable:
    """
    A variable in a review template.

    Attributes:
        name: Variable name (e.g., "reviewer_name")
        var_type: Type of the variable
        description: Description of the variable
        default: Default value
        required: Whether the variable is required
        options: Valid options for enum-like variables
    """
    name: str
    var_type: VariableType
    description: str = ""
    default: Any = None
    required: bool = False
    options: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "var_type": self.var_type.value,
            "description": self.description,
            "default": self.default,
            "required": self.required,
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateVariable":
        """Create from dictionary."""
        data = data.copy()
        data["var_type"] = VariableType(data["var_type"])
        return cls(**data)


@dataclass
class TemplateSection:
    """A section within a review template."""
    name: str
    title: str
    content: str
    order: int = 0
    conditional: Optional[str] = None  # Condition for including section

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemplateSection":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ReviewTemplate:
    """
    A review template.

    Attributes:
        template_id: Unique identifier
        name: Template name
        description: Template description
        category: Template category
        sections: Template sections
        variables: Template variables
        created_at: Creation timestamp
        updated_at: Last update timestamp
        author: Template author
        version: Template version
        tags: Associated tags
    """
    template_id: str
    name: str
    description: str
    category: TemplateCategory
    sections: List[TemplateSection] = field(default_factory=list)
    variables: List[TemplateVariable] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    author: Optional[str] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "sections": [s.to_dict() for s in self.sections],
            "variables": [v.to_dict() for v in self.variables],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
            "version": self.version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewTemplate":
        """Create from dictionary."""
        data = data.copy()
        data["category"] = TemplateCategory(data["category"])
        data["sections"] = [TemplateSection.from_dict(s) for s in data.get("sections", [])]
        data["variables"] = [TemplateVariable.from_dict(v) for v in data.get("variables", [])]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def render(self, values: Dict[str, Any]) -> str:
        """
        Render the template with provided values.

        Args:
            values: Variable values

        Returns:
            Rendered template content
        """
        # Add default values
        for var in self.variables:
            if var.name not in values and var.default is not None:
                values[var.name] = var.default

        lines = []
        for section in sorted(self.sections, key=lambda s: s.order):
            # Check conditional
            if section.conditional:
                cond_var = section.conditional.lstrip("!")
                cond_val = values.get(cond_var, False)
                if section.conditional.startswith("!"):
                    cond_val = not cond_val
                if not cond_val:
                    continue

            # Render section
            content = section.content
            for var_name, var_value in values.items():
                placeholder = f"{{{{{var_name}}}}}"
                if isinstance(var_value, list):
                    var_value = ", ".join(str(v) for v in var_value)
                content = content.replace(placeholder, str(var_value))

            lines.extend([
                f"## {section.title}",
                "",
                content,
                "",
            ])

        return "\n".join(lines)


# ============================================================================
# Built-in Templates
# ============================================================================

BUILTIN_TEMPLATES = [
    ReviewTemplate(
        template_id="general-review",
        name="General Code Review",
        description="Standard template for general code review",
        category=TemplateCategory.GENERAL,
        sections=[
            TemplateSection(
                name="summary",
                title="Review Summary",
                content="**Reviewer:** {{reviewer_name}}\n**Date:** {{review_date}}\n**Decision:** {{decision}}\n\n{{summary}}",
                order=1,
            ),
            TemplateSection(
                name="changes",
                title="Changes Reviewed",
                content="Files reviewed: {{file_count}}\n\n{{files_list}}",
                order=2,
            ),
            TemplateSection(
                name="findings",
                title="Findings",
                content="Total issues: {{issue_count}}\n\n{{findings}}",
                order=3,
            ),
            TemplateSection(
                name="recommendations",
                title="Recommendations",
                content="{{recommendations}}",
                order=4,
                conditional="has_recommendations",
            ),
        ],
        variables=[
            TemplateVariable(name="reviewer_name", var_type=VariableType.STRING, required=True),
            TemplateVariable(name="review_date", var_type=VariableType.DATE, required=True),
            TemplateVariable(name="decision", var_type=VariableType.STRING, required=True,
                           options=["approve", "request_changes", "comment"]),
            TemplateVariable(name="summary", var_type=VariableType.STRING),
            TemplateVariable(name="file_count", var_type=VariableType.NUMBER, default=0),
            TemplateVariable(name="files_list", var_type=VariableType.STRING),
            TemplateVariable(name="issue_count", var_type=VariableType.NUMBER, default=0),
            TemplateVariable(name="findings", var_type=VariableType.STRING),
            TemplateVariable(name="has_recommendations", var_type=VariableType.BOOLEAN, default=False),
            TemplateVariable(name="recommendations", var_type=VariableType.STRING),
        ],
        tags=["general", "standard"],
    ),
    ReviewTemplate(
        template_id="security-review",
        name="Security Review",
        description="Template focused on security analysis",
        category=TemplateCategory.SECURITY,
        sections=[
            TemplateSection(
                name="summary",
                title="Security Review Summary",
                content="**Risk Level:** {{risk_level}}\n**Vulnerabilities Found:** {{vuln_count}}\n\n{{summary}}",
                order=1,
            ),
            TemplateSection(
                name="critical",
                title="Critical Vulnerabilities",
                content="{{critical_vulns}}",
                order=2,
                conditional="has_critical",
            ),
            TemplateSection(
                name="high",
                title="High Severity Issues",
                content="{{high_vulns}}",
                order=3,
                conditional="has_high",
            ),
            TemplateSection(
                name="other",
                title="Other Security Issues",
                content="{{other_vulns}}",
                order=4,
            ),
            TemplateSection(
                name="compliance",
                title="Compliance Notes",
                content="{{compliance_notes}}",
                order=5,
                conditional="has_compliance",
            ),
        ],
        variables=[
            TemplateVariable(name="risk_level", var_type=VariableType.STRING, required=True,
                           options=["critical", "high", "medium", "low"]),
            TemplateVariable(name="vuln_count", var_type=VariableType.NUMBER, default=0),
            TemplateVariable(name="summary", var_type=VariableType.STRING),
            TemplateVariable(name="has_critical", var_type=VariableType.BOOLEAN, default=False),
            TemplateVariable(name="critical_vulns", var_type=VariableType.STRING),
            TemplateVariable(name="has_high", var_type=VariableType.BOOLEAN, default=False),
            TemplateVariable(name="high_vulns", var_type=VariableType.STRING),
            TemplateVariable(name="other_vulns", var_type=VariableType.STRING),
            TemplateVariable(name="has_compliance", var_type=VariableType.BOOLEAN, default=False),
            TemplateVariable(name="compliance_notes", var_type=VariableType.STRING),
        ],
        tags=["security", "vulnerability", "compliance"],
    ),
    ReviewTemplate(
        template_id="pr-checklist",
        name="PR Review Checklist",
        description="Checklist-based PR review template",
        category=TemplateCategory.GENERAL,
        sections=[
            TemplateSection(
                name="code_quality",
                title="Code Quality Checklist",
                content="- [ ] Code follows project style guide\n- [ ] No unnecessary complexity\n- [ ] Proper error handling\n- [ ] No hardcoded values\n- [ ] Meaningful variable/function names",
                order=1,
            ),
            TemplateSection(
                name="testing",
                title="Testing Checklist",
                content="- [ ] Unit tests added/updated\n- [ ] Tests pass locally\n- [ ] Edge cases covered\n- [ ] Test coverage maintained",
                order=2,
            ),
            TemplateSection(
                name="documentation",
                title="Documentation Checklist",
                content="- [ ] Code is self-documenting\n- [ ] Complex logic documented\n- [ ] API changes documented\n- [ ] README updated if needed",
                order=3,
            ),
            TemplateSection(
                name="security",
                title="Security Checklist",
                content="- [ ] No sensitive data exposed\n- [ ] Input validation present\n- [ ] No SQL injection risks\n- [ ] Dependencies are secure",
                order=4,
            ),
        ],
        variables=[],
        tags=["checklist", "pr", "standard"],
    ),
]


# ============================================================================
# Template Store
# ============================================================================

class TemplateStore:
    """Persistent store for templates."""

    def __init__(self, store_path: Path):
        """
        Initialize the template store.

        Args:
            store_path: Path to the data file
        """
        self.store_path = store_path
        self._ensure_store()

    def _ensure_store(self) -> None:
        """Ensure store file exists with built-in templates."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._write_store({
                "templates": [t.to_dict() for t in BUILTIN_TEMPLATES],
                "version": 1,
            })

    def _read_store(self) -> Dict[str, Any]:
        """Read store with file locking."""
        with open(self.store_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_store(self, data: Dict[str, Any]) -> None:
        """Write store with file locking."""
        with open(self.store_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def add(self, template: ReviewTemplate) -> None:
        """Add a template."""
        data = self._read_store()
        # Remove existing with same ID
        data["templates"] = [t for t in data["templates"] if t["template_id"] != template.template_id]
        data["templates"].append(template.to_dict())
        self._write_store(data)

    def get(self, template_id: str) -> Optional[ReviewTemplate]:
        """Get a template by ID."""
        data = self._read_store()
        for t in data["templates"]:
            if t["template_id"] == template_id:
                return ReviewTemplate.from_dict(t)
        return None

    def get_all(
        self,
        category: Optional[TemplateCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ReviewTemplate]:
        """Get templates with optional filters."""
        data = self._read_store()
        templates = []

        for t_data in data["templates"]:
            if category and t_data["category"] != category.value:
                continue
            if tags and not any(tag in t_data.get("tags", []) for tag in tags):
                continue
            templates.append(ReviewTemplate.from_dict(t_data))

        return templates

    def delete(self, template_id: str) -> bool:
        """Delete a template."""
        data = self._read_store()
        original_len = len(data["templates"])
        data["templates"] = [t for t in data["templates"] if t["template_id"] != template_id]
        if len(data["templates"]) < original_len:
            self._write_store(data)
            return True
        return False


# ============================================================================
# Template Manager
# ============================================================================

class TemplateManager:
    """
    Manages review templates.

    Example:
        manager = TemplateManager()

        # Get a template
        template = manager.get_template("general-review")

        # Render with values
        content = manager.apply_template(template, {
            "reviewer_name": "John",
            "review_date": "2024-01-15",
            "decision": "approve",
        })
    """

    BUS_TOPICS = {
        "apply": "review.template.apply",
        "create": "review.template.create",
    }

    def __init__(
        self,
        bus_path: Optional[Path] = None,
        store_path: Optional[Path] = None,
    ):
        """
        Initialize the template manager.

        Args:
            bus_path: Path to event bus file
            store_path: Path to template store
        """
        self.bus_path = bus_path or self._get_bus_path()
        self.store = TemplateStore(store_path or self._get_store_path())

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_store_path(self) -> Path:
        """Get path to template store."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        data_dir = pluribus_root / ".pluribus" / "review" / "data"
        return data_dir / "review_templates.json"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "template") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "template-manager",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def get_template(self, template_id: str) -> Optional[ReviewTemplate]:
        """
        Get a template by ID.

        Args:
            template_id: Template ID

        Returns:
            Template or None if not found
        """
        return self.store.get(template_id)

    def list_templates(
        self,
        category: Optional[TemplateCategory] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ReviewTemplate]:
        """
        List available templates.

        Args:
            category: Filter by category
            tags: Filter by tags

        Returns:
            List of matching templates
        """
        return self.store.get_all(category=category, tags=tags)

    def create_template(
        self,
        name: str,
        description: str,
        category: TemplateCategory,
        sections: List[TemplateSection],
        variables: Optional[List[TemplateVariable]] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ReviewTemplate:
        """
        Create a new template.

        Args:
            name: Template name
            description: Template description
            category: Template category
            sections: Template sections
            variables: Template variables
            author: Template author
            tags: Associated tags

        Returns:
            Created template

        Emits:
            review.template.create
        """
        template_id = str(uuid.uuid4())[:8]

        template = ReviewTemplate(
            template_id=template_id,
            name=name,
            description=description,
            category=category,
            sections=sections,
            variables=variables or [],
            author=author,
            tags=tags or [],
        )

        self.store.add(template)

        self._emit_event(self.BUS_TOPICS["create"], {
            "template_id": template_id,
            "name": name,
            "category": category.value,
        })

        return template

    def apply_template(
        self,
        template: ReviewTemplate,
        values: Dict[str, Any],
    ) -> str:
        """
        Apply a template with provided values.

        Args:
            template: Template to apply
            values: Variable values

        Returns:
            Rendered template content

        Emits:
            review.template.apply
        """
        # Validate required variables
        missing = []
        for var in template.variables:
            if var.required and var.name not in values:
                missing.append(var.name)

        if missing:
            raise ValueError(f"Missing required variables: {', '.join(missing)}")

        content = template.render(values)

        self._emit_event(self.BUS_TOPICS["apply"], {
            "template_id": template.template_id,
            "template_name": template.name,
            "values_provided": list(values.keys()),
        })

        return content

    def apply_template_by_id(
        self,
        template_id: str,
        values: Dict[str, Any],
    ) -> str:
        """
        Apply a template by ID.

        Args:
            template_id: Template ID
            values: Variable values

        Returns:
            Rendered template content
        """
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        return self.apply_template(template, values)

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template.

        Args:
            template_id: Template ID

        Returns:
            True if deleted
        """
        return self.store.delete(template_id)

    def duplicate_template(
        self,
        template_id: str,
        new_name: Optional[str] = None,
    ) -> ReviewTemplate:
        """
        Duplicate an existing template.

        Args:
            template_id: Source template ID
            new_name: Name for the new template

        Returns:
            New template
        """
        source = self.get_template(template_id)
        if not source:
            raise ValueError(f"Template not found: {template_id}")

        new_id = str(uuid.uuid4())[:8]
        new_template = ReviewTemplate(
            template_id=new_id,
            name=new_name or f"{source.name} (Copy)",
            description=source.description,
            category=source.category,
            sections=source.sections.copy(),
            variables=source.variables.copy(),
            author=source.author,
            tags=source.tags.copy(),
        )

        self.store.add(new_template)
        return new_template


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Template Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Template Manager (Step 175)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List templates")
    list_parser.add_argument("--category", choices=[c.value for c in TemplateCategory])

    # Show command
    show_parser = subparsers.add_parser("show", help="Show template details")
    show_parser.add_argument("template_id")

    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply template")
    apply_parser.add_argument("template_id")
    apply_parser.add_argument("--values", help="JSON values or @file")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    manager = TemplateManager()

    if args.command == "list":
        category = TemplateCategory(args.category) if args.category else None
        templates = manager.list_templates(category=category)

        if args.json:
            print(json.dumps([t.to_dict() for t in templates], indent=2))
        else:
            print(f"Found {len(templates)} templates:")
            for t in templates:
                print(f"  [{t.template_id}] {t.name}")
                print(f"    Category: {t.category.value} | Tags: {', '.join(t.tags)}")

    elif args.command == "show":
        template = manager.get_template(args.template_id)
        if not template:
            print(f"Template not found: {args.template_id}")
            return 1

        if args.json:
            print(json.dumps(template.to_dict(), indent=2))
        else:
            print(f"Template: {template.name}")
            print(f"  ID: {template.template_id}")
            print(f"  Category: {template.category.value}")
            print(f"  Description: {template.description}")
            print(f"\nSections:")
            for s in template.sections:
                print(f"  - {s.title}")
            print(f"\nVariables:")
            for v in template.variables:
                req = "(required)" if v.required else ""
                print(f"  - {v.name}: {v.var_type.value} {req}")

    elif args.command == "apply":
        values = {}
        if args.values:
            if args.values.startswith("@"):
                with open(args.values[1:]) as f:
                    values = json.load(f)
            else:
                values = json.loads(args.values)

        # Add default date
        if "review_date" not in values:
            values["review_date"] = datetime.now().strftime("%Y-%m-%d")

        try:
            content = manager.apply_template_by_id(args.template_id, values)
            print(content)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
