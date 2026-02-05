#!/usr/bin/env python3
"""
template_engine.py - Code Template Engine (Step 72)

PBTSO Phase: PLAN, ITERATE

Provides:
- Template definition and storage
- Variable substitution with type checking
- Conditional sections in templates
- Template inheritance and composition
- Multi-language template support

Bus Topics:
- code.template.render
- code.template.create
- code.template.error

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class VariableType(Enum):
    """Types for template variables."""
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    CODE = "code"


@dataclass
class TemplateVariable:
    """Definition of a template variable."""
    name: str
    var_type: VariableType = VariableType.STRING
    default: Optional[Any] = None
    required: bool = False
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None

    def validate(self, value: Any) -> bool:
        """Validate value against variable definition."""
        if value is None:
            return not self.required

        type_checks = {
            VariableType.STRING: lambda v: isinstance(v, str),
            VariableType.INTEGER: lambda v: isinstance(v, int),
            VariableType.BOOLEAN: lambda v: isinstance(v, bool),
            VariableType.LIST: lambda v: isinstance(v, list),
            VariableType.DICT: lambda v: isinstance(v, dict),
            VariableType.CODE: lambda v: isinstance(v, str),
        }

        if not type_checks.get(self.var_type, lambda v: True)(value):
            return False

        if self.validator:
            return self.validator(value)

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.var_type.value,
            "default": self.default,
            "required": self.required,
            "description": self.description,
        }


@dataclass
class TemplateConfig:
    """Configuration for the template engine."""
    template_dir: str = "/pluribus/pia/oagent/code/templates"
    enable_caching: bool = True
    cache_ttl_s: int = 3600
    strict_variables: bool = True
    allow_undefined: bool = False
    auto_escape: bool = False
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_dir": self.template_dir,
            "enable_caching": self.enable_caching,
            "cache_ttl_s": self.cache_ttl_s,
            "strict_variables": self.strict_variables,
            "allow_undefined": self.allow_undefined,
        }


# =============================================================================
# Types
# =============================================================================

@dataclass
class CodeTemplate:
    """A code template definition."""
    id: str
    name: str
    language: str
    content: str
    variables: List[TemplateVariable] = field(default_factory=list)
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    parent: Optional[str] = None  # Parent template for inheritance
    blocks: Dict[str, str] = field(default_factory=dict)  # Named blocks for override
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "content": self.content,
            "variables": [v.to_dict() for v in self.variables],
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "parent": self.parent,
            "blocks": self.blocks,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeTemplate":
        """Create template from dictionary."""
        variables = [
            TemplateVariable(
                name=v["name"],
                var_type=VariableType(v.get("type", "string")),
                default=v.get("default"),
                required=v.get("required", False),
                description=v.get("description", ""),
            )
            for v in data.get("variables", [])
        ]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            language=data["language"],
            content=data["content"],
            variables=variables,
            description=data.get("description", ""),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            parent=data.get("parent"),
            blocks=data.get("blocks", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


@dataclass
class TemplateResult:
    """Result of template rendering."""
    success: bool
    content: str
    template_id: str
    variables_used: List[str]
    elapsed_ms: float
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "template_id": self.template_id,
            "variables_used": self.variables_used,
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
# Template Engine
# =============================================================================

class TemplateEngine:
    """
    Code template engine.

    PBTSO Phase: PLAN, ITERATE

    Responsibilities:
    - Manage template definitions
    - Render templates with variable substitution
    - Support template inheritance
    - Handle conditional sections
    - Cache rendered templates

    Template Syntax:
    - Variables: {{variable_name}}
    - Conditionals: {% if condition %}...{% endif %}
    - Loops: {% for item in list %}...{% endfor %}
    - Blocks: {% block name %}...{% endblock %}
    - Includes: {% include "template_name" %}

    Usage:
        engine = TemplateEngine(config)
        result = engine.render("template_name", {"var": "value"})
    """

    BUS_TOPICS = {
        "render": "code.template.render",
        "create": "code.template.create",
        "error": "code.template.error",
        "heartbeat": "code.template.heartbeat",
    }

    # Regex patterns for template syntax
    VAR_PATTERN = re.compile(r"\{\{\s*(\w+)\s*\}\}")
    CONDITIONAL_PATTERN = re.compile(
        r"\{%\s*if\s+(not\s+)?(\w+)\s*%\}(.*?)\{%\s*endif\s*%\}",
        re.DOTALL
    )
    FOR_PATTERN = re.compile(
        r"\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}",
        re.DOTALL
    )
    BLOCK_PATTERN = re.compile(
        r"\{%\s*block\s+(\w+)\s*%\}(.*?)\{%\s*endblock\s*%\}",
        re.DOTALL
    )
    INCLUDE_PATTERN = re.compile(r'\{%\s*include\s+"([^"]+)"\s*%\}')

    # Built-in templates
    BUILTIN_TEMPLATES: Dict[str, CodeTemplate] = {}

    def __init__(
        self,
        config: Optional[TemplateConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or TemplateConfig()
        self.bus = bus or LockedAgentBus()
        self._templates: Dict[str, CodeTemplate] = {}
        self._cache: Dict[str, tuple[str, float]] = {}

        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load built-in templates."""
        builtins = [
            CodeTemplate(
                id="python-function",
                name="Python Function",
                language="python",
                content='''def {{name}}({{params}}) -> {{return_type}}:
    """
    {{docstring}}

    Args:
{% for param in param_docs %}
        {{param}}
{% endfor %}

    Returns:
        {{return_doc}}
    """
{% if has_body %}
    {{body}}
{% endif %}
{% if not has_body %}
    pass
{% endif %}
''',
                variables=[
                    TemplateVariable("name", VariableType.STRING, required=True),
                    TemplateVariable("params", VariableType.STRING, default=""),
                    TemplateVariable("return_type", VariableType.STRING, default="None"),
                    TemplateVariable("docstring", VariableType.STRING, default="TODO: Document"),
                    TemplateVariable("param_docs", VariableType.LIST, default=[]),
                    TemplateVariable("return_doc", VariableType.STRING, default="None"),
                    TemplateVariable("has_body", VariableType.BOOLEAN, default=False),
                    TemplateVariable("body", VariableType.CODE, default="pass"),
                ],
                category="function",
                tags=["python", "function"],
            ),
            CodeTemplate(
                id="python-class",
                name="Python Class",
                language="python",
                content='''class {{name}}{% if bases %}({{bases}}){% endif %}:
    """
    {{docstring}}
    """

    def __init__(self{{init_params}}):
        """Initialize {{name}}."""
{% for attr in attributes %}
        self.{{attr}} = {{attr}}
{% endfor %}
{% if not attributes %}
        pass
{% endif %}

{% for method in methods %}
{{method}}

{% endfor %}
''',
                variables=[
                    TemplateVariable("name", VariableType.STRING, required=True),
                    TemplateVariable("bases", VariableType.STRING, default=""),
                    TemplateVariable("docstring", VariableType.STRING, default="TODO: Document"),
                    TemplateVariable("init_params", VariableType.STRING, default=""),
                    TemplateVariable("attributes", VariableType.LIST, default=[]),
                    TemplateVariable("methods", VariableType.LIST, default=[]),
                ],
                category="class",
                tags=["python", "class"],
            ),
            CodeTemplate(
                id="python-dataclass",
                name="Python Dataclass",
                language="python",
                content='''from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class {{name}}:
    """
    {{docstring}}
    """
{% for field in fields %}
    {{field}}
{% endfor %}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
{% for attr in attributes %}
            "{{attr}}": self.{{attr}},
{% endfor %}
        }
''',
                variables=[
                    TemplateVariable("name", VariableType.STRING, required=True),
                    TemplateVariable("docstring", VariableType.STRING, default="TODO: Document"),
                    TemplateVariable("fields", VariableType.LIST, default=[]),
                    TemplateVariable("attributes", VariableType.LIST, default=[]),
                ],
                category="dataclass",
                tags=["python", "dataclass"],
            ),
            CodeTemplate(
                id="python-test",
                name="Python Test",
                language="python",
                content='''import pytest

class Test{{class_name}}:
    """Tests for {{class_name}}."""

{% for test in tests %}
    def test_{{test.name}}(self):
        """Test {{test.description}}."""
        # Arrange
        {{test.arrange}}

        # Act
        {{test.act}}

        # Assert
        {{test.assertions}}

{% endfor %}
''',
                variables=[
                    TemplateVariable("class_name", VariableType.STRING, required=True),
                    TemplateVariable("tests", VariableType.LIST, default=[]),
                ],
                category="test",
                tags=["python", "test", "pytest"],
            ),
            CodeTemplate(
                id="typescript-interface",
                name="TypeScript Interface",
                language="typescript",
                content='''/**
 * {{docstring}}
 */
export interface {{name}} {
{% for field in fields %}
    {{field.name}}{% if not field.required %}?{% endif %}: {{field.type}};
{% endfor %}
}
''',
                variables=[
                    TemplateVariable("name", VariableType.STRING, required=True),
                    TemplateVariable("docstring", VariableType.STRING, default="TODO: Document"),
                    TemplateVariable("fields", VariableType.LIST, default=[]),
                ],
                category="interface",
                tags=["typescript", "interface"],
            ),
        ]

        for template in builtins:
            self._templates[template.id] = template

    def register_template(self, template: CodeTemplate) -> str:
        """
        Register a new template.

        Args:
            template: Template to register

        Returns:
            Template ID
        """
        self._templates[template.id] = template

        self.bus.emit({
            "topic": self.BUS_TOPICS["create"],
            "kind": "template",
            "actor": "template-engine",
            "data": {
                "template_id": template.id,
                "name": template.name,
                "language": template.language,
            },
        })

        return template.id

    def get_template(self, template_id: str) -> Optional[CodeTemplate]:
        """Get template by ID."""
        return self._templates.get(template_id)

    def list_templates(
        self,
        language: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[CodeTemplate]:
        """List templates with optional filtering."""
        templates = list(self._templates.values())

        if language:
            templates = [t for t in templates if t.language == language]

        if category:
            templates = [t for t in templates if t.category == category]

        if tags:
            tag_set = set(tags)
            templates = [t for t in templates if tag_set.intersection(t.tags)]

        return templates

    def render(
        self,
        template_id: str,
        variables: Dict[str, Any],
        block_overrides: Optional[Dict[str, str]] = None,
    ) -> TemplateResult:
        """
        Render a template with given variables.

        Args:
            template_id: ID of template to render
            variables: Variable values for substitution
            block_overrides: Optional block content overrides

        Returns:
            TemplateResult with rendered content
        """
        start_time = time.time()
        variables_used: List[str] = []
        warnings: List[str] = []

        template = self._templates.get(template_id)
        if not template:
            return TemplateResult(
                success=False,
                content="",
                template_id=template_id,
                variables_used=[],
                elapsed_ms=(time.time() - start_time) * 1000,
                error=f"Template not found: {template_id}",
            )

        try:
            # Validate variables
            for var in template.variables:
                value = variables.get(var.name, var.default)
                if not var.validate(value):
                    if self.config.strict_variables:
                        raise ValueError(f"Invalid value for variable: {var.name}")
                    warnings.append(f"Invalid value for variable: {var.name}")

                if var.name in variables:
                    variables_used.append(var.name)

            # Build context with defaults
            context = {v.name: v.default for v in template.variables}
            context.update(variables)

            # Handle template inheritance
            content = template.content
            if template.parent:
                parent = self._templates.get(template.parent)
                if parent:
                    content = self._merge_with_parent(parent.content, content, block_overrides)

            # Apply block overrides
            if block_overrides:
                content = self._apply_block_overrides(content, block_overrides)

            # Render content
            rendered = self._render_content(content, context)

            # Emit render event
            self.bus.emit({
                "topic": self.BUS_TOPICS["render"],
                "kind": "template",
                "actor": "template-engine",
                "data": {
                    "template_id": template_id,
                    "variables_count": len(variables_used),
                },
            })

            return TemplateResult(
                success=True,
                content=rendered,
                template_id=template_id,
                variables_used=variables_used,
                elapsed_ms=(time.time() - start_time) * 1000,
                warnings=warnings,
            )

        except Exception as e:
            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "error",
                "actor": "template-engine",
                "data": {
                    "template_id": template_id,
                    "error": str(e),
                },
            })

            return TemplateResult(
                success=False,
                content="",
                template_id=template_id,
                variables_used=variables_used,
                elapsed_ms=(time.time() - start_time) * 1000,
                warnings=warnings,
                error=str(e),
            )

    def _render_content(self, content: str, context: Dict[str, Any]) -> str:
        """Render template content with context."""
        # Process includes first
        content = self._process_includes(content, context)

        # Process for loops
        content = self._process_for_loops(content, context)

        # Process conditionals
        content = self._process_conditionals(content, context)

        # Process variables
        content = self._process_variables(content, context)

        return content

    def _process_includes(self, content: str, context: Dict[str, Any]) -> str:
        """Process include directives."""
        def replace_include(match):
            template_name = match.group(1)
            template = self._templates.get(template_name)
            if template:
                return self._render_content(template.content, context)
            return f"<!-- Template not found: {template_name} -->"

        return self.INCLUDE_PATTERN.sub(replace_include, content)

    def _process_conditionals(self, content: str, context: Dict[str, Any]) -> str:
        """Process conditional directives."""
        def replace_conditional(match):
            negate_group = match.group(1)  # "not " or None
            var_name = match.group(2)
            body = match.group(3)

            # Handle negation (not var)
            negate = negate_group is not None

            value = context.get(var_name, False)
            condition = bool(value)
            if negate:
                condition = not condition

            if condition:
                return body
            return ""

        return self.CONDITIONAL_PATTERN.sub(replace_conditional, content)

    def _process_for_loops(self, content: str, context: Dict[str, Any]) -> str:
        """Process for loop directives."""
        def replace_loop(match):
            item_name = match.group(1)
            list_name = match.group(2)
            body = match.group(3)

            items = context.get(list_name, [])
            if not isinstance(items, list):
                return ""

            result = []
            for item in items:
                loop_context = context.copy()
                loop_context[item_name] = item
                rendered_body = self._process_variables(body, loop_context)
                # Handle nested access for dicts
                if isinstance(item, dict):
                    for key, value in item.items():
                        rendered_body = rendered_body.replace(
                            f"{{{{{item_name}.{key}}}}}",
                            str(value)
                        )
                result.append(rendered_body)

            return "".join(result)

        return self.FOR_PATTERN.sub(replace_loop, content)

    def _process_variables(self, content: str, context: Dict[str, Any]) -> str:
        """Process variable substitutions."""
        def replace_var(match):
            var_name = match.group(1)
            value = context.get(var_name)

            if value is None:
                if self.config.allow_undefined:
                    return f"{{{{ {var_name} }}}}"
                return ""

            return str(value)

        return self.VAR_PATTERN.sub(replace_var, content)

    def _merge_with_parent(
        self,
        parent_content: str,
        child_content: str,
        overrides: Optional[Dict[str, str]] = None,
    ) -> str:
        """Merge child template with parent."""
        # Extract blocks from child
        child_blocks = dict(self.BLOCK_PATTERN.findall(child_content))
        if overrides:
            child_blocks.update(overrides)

        # Replace parent blocks with child blocks
        def replace_block(match):
            block_name = match.group(1)
            default_content = match.group(2)
            return child_blocks.get(block_name, default_content)

        return self.BLOCK_PATTERN.sub(replace_block, parent_content)

    def _apply_block_overrides(
        self,
        content: str,
        overrides: Dict[str, str],
    ) -> str:
        """Apply block overrides to content."""
        def replace_block(match):
            block_name = match.group(1)
            default_content = match.group(2)
            return overrides.get(block_name, default_content)

        return self.BLOCK_PATTERN.sub(replace_block, content)

    def create_from_code(
        self,
        name: str,
        code: str,
        language: str,
        variables: Optional[List[str]] = None,
    ) -> CodeTemplate:
        """
        Create a template from existing code.

        Detects variables in the code and creates template.
        """
        # Find potential variables (words that look like placeholders)
        found_vars = set(self.VAR_PATTERN.findall(code))
        if variables:
            found_vars.update(variables)

        template_vars = [
            TemplateVariable(name=var, var_type=VariableType.STRING)
            for var in found_vars
        ]

        template = CodeTemplate(
            id=f"custom-{uuid.uuid4().hex[:8]}",
            name=name,
            language=language,
            content=code,
            variables=template_vars,
            category="custom",
        )

        self.register_template(template)
        return template

    def get_stats(self) -> Dict[str, Any]:
        """Get template engine statistics."""
        return {
            "template_count": len(self._templates),
            "languages": list(set(t.language for t in self._templates.values())),
            "categories": list(set(t.category for t in self._templates.values())),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Template Engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Template Engine (Step 72)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # render command
    render_parser = subparsers.add_parser("render", help="Render a template")
    render_parser.add_argument("template_id", help="Template ID")
    render_parser.add_argument("--var", "-v", nargs=2, action="append", metavar=("NAME", "VALUE"),
                               help="Variable value")
    render_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List templates")
    list_parser.add_argument("--language", "-l", help="Filter by language")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # show command
    show_parser = subparsers.add_parser("show", help="Show template details")
    show_parser.add_argument("template_id", help="Template ID")
    show_parser.add_argument("--json", action="store_true", help="JSON output")

    # stats command
    subparsers.add_parser("stats", help="Show engine stats")

    args = parser.parse_args()

    engine = TemplateEngine()

    if args.command == "render":
        variables = {}
        if args.var:
            for name, value in args.var:
                # Try to parse as JSON, fall back to string
                try:
                    variables[name] = json.loads(value)
                except json.JSONDecodeError:
                    variables[name] = value

        result = engine.render(args.template_id, variables)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(result.content)
            else:
                print(f"Error: {result.error}")
                return 1
        return 0

    elif args.command == "list":
        templates = engine.list_templates(
            language=args.language,
            category=args.category,
        )

        if args.json:
            print(json.dumps([t.to_dict() for t in templates], indent=2))
        else:
            for t in templates:
                print(f"{t.id}: {t.name} ({t.language}) - {t.category}")
        return 0

    elif args.command == "show":
        template = engine.get_template(args.template_id)

        if not template:
            print(f"Template not found: {args.template_id}")
            return 1

        if args.json:
            print(json.dumps(template.to_dict(), indent=2))
        else:
            print(f"ID: {template.id}")
            print(f"Name: {template.name}")
            print(f"Language: {template.language}")
            print(f"Category: {template.category}")
            print(f"Variables:")
            for v in template.variables:
                req = " (required)" if v.required else ""
                print(f"  - {v.name}: {v.var_type.value}{req}")
            print(f"\nContent:\n{template.content}")
        return 0

    elif args.command == "stats":
        stats = engine.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
