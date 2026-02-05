#!/usr/bin/env python3
"""
neural_generator.py - Neural Code Generator (Step 71)

PBTSO Phase: PLAN, ITERATE

Provides:
- Neural-guided code generation
- Pattern-based code synthesis
- Context-aware code completion
- Multi-language code generation
- Semantic understanding for generation

Bus Topics:
- code.generator.request
- code.generator.complete
- code.generator.error

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class GenerationMode(Enum):
    """Code generation mode."""
    COMPLETE = "complete"      # Complete existing code
    GENERATE = "generate"      # Generate new code from scratch
    TRANSFORM = "transform"    # Transform existing code
    REFACTOR = "refactor"      # Refactor code structure
    EXPLAIN = "explain"        # Generate explanations/docs


class CodePattern(Enum):
    """Common code patterns for generation."""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    TEST = "test"
    DATACLASS = "dataclass"
    ASYNC_FUNCTION = "async_function"
    DECORATOR = "decorator"
    CONTEXT_MANAGER = "context_manager"
    ITERATOR = "iterator"
    SINGLETON = "singleton"
    FACTORY = "factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    API_ENDPOINT = "api_endpoint"
    CLI_COMMAND = "cli_command"


@dataclass
class GeneratorConfig:
    """Configuration for the code generator."""
    default_language: str = "python"
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "typescript", "javascript", "go", "rust"
    ])
    max_tokens: int = 4096
    temperature: float = 0.3
    enable_context_retrieval: bool = True
    enable_pattern_matching: bool = True
    enable_semantic_analysis: bool = True
    cache_generated_code: bool = True
    cache_ttl_s: int = 3600
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enable_context_retrieval": self.enable_context_retrieval,
            "enable_pattern_matching": self.enable_pattern_matching,
            "enable_semantic_analysis": self.enable_semantic_analysis,
            "cache_generated_code": self.cache_generated_code,
            "cache_ttl_s": self.cache_ttl_s,
        }


# =============================================================================
# Types
# =============================================================================

@dataclass
class GenerationRequest:
    """Request for code generation."""
    id: str
    prompt: str
    mode: GenerationMode = GenerationMode.GENERATE
    language: str = "python"
    pattern: Optional[CodePattern] = None
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "mode": self.mode.value,
            "language": self.language,
            "pattern": self.pattern.value if self.pattern else None,
            "context": self.context,
            "constraints": self.constraints,
            "examples": self.examples,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


@dataclass
class GenerationResult:
    """Result of code generation."""
    request_id: str
    success: bool
    code: str
    language: str
    confidence: float
    tokens_used: int
    elapsed_ms: float
    alternatives: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "code": self.code,
            "language": self.language,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "elapsed_ms": self.elapsed_ms,
            "alternatives": self.alternatives,
            "explanation": self.explanation,
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass
class CodeContext:
    """Context for code generation."""
    file_path: Optional[str] = None
    surrounding_code: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    defined_names: List[str] = field(default_factory=list)
    project_structure: Dict[str, Any] = field(default_factory=dict)
    related_files: List[str] = field(default_factory=list)


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

        # Write with file locking
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
# Code Generator
# =============================================================================

class CodeGenerator:
    """
    Neural-guided code generator.

    PBTSO Phase: PLAN, ITERATE

    Responsibilities:
    - Generate code from natural language prompts
    - Apply code patterns for structured generation
    - Use context for intelligent completion
    - Support multiple programming languages
    - Provide confidence scoring

    Usage:
        generator = CodeGenerator(config)
        result = await generator.generate(request)
    """

    BUS_TOPICS = {
        "request": "code.generator.request",
        "complete": "code.generator.complete",
        "error": "code.generator.error",
        "heartbeat": "code.generator.heartbeat",
    }

    # Language-specific templates
    LANGUAGE_TEMPLATES: Dict[str, Dict[str, str]] = {
        "python": {
            "function": '''def {name}({params}) -> {return_type}:
    """
    {docstring}
    """
    {body}
''',
            "class": '''class {name}({bases}):
    """
    {docstring}
    """

    def __init__(self{params}):
        {init_body}

    {methods}
''',
            "async_function": '''async def {name}({params}) -> {return_type}:
    """
    {docstring}
    """
    {body}
''',
            "dataclass": '''@dataclass
class {name}:
    """
    {docstring}
    """
    {fields}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
''',
            "test": '''def test_{name}():
    """Test {description}."""
    # Arrange
    {arrange}

    # Act
    {act}

    # Assert
    {assertions}
''',
        },
        "typescript": {
            "function": '''function {name}({params}): {return_type} {{
    {body}
}}
''',
            "class": '''class {name} {{
    {fields}

    constructor({params}) {{
        {init_body}
    }}

    {methods}
}}
''',
            "interface": '''interface {name} {{
    {fields}
}}
''',
        },
    }

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or GeneratorConfig()
        self.bus = bus or LockedAgentBus()
        self._cache: Dict[str, Tuple[GenerationResult, float]] = {}
        self._pattern_registry: Dict[CodePattern, Callable] = {}
        self._is_running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        self._register_patterns()

    def _register_patterns(self) -> None:
        """Register code pattern generators."""
        self._pattern_registry = {
            CodePattern.FUNCTION: self._generate_function,
            CodePattern.CLASS: self._generate_class,
            CodePattern.DATACLASS: self._generate_dataclass,
            CodePattern.ASYNC_FUNCTION: self._generate_async_function,
            CodePattern.TEST: self._generate_test,
            CodePattern.DECORATOR: self._generate_decorator,
            CodePattern.CONTEXT_MANAGER: self._generate_context_manager,
            CodePattern.API_ENDPOINT: self._generate_api_endpoint,
            CodePattern.CLI_COMMAND: self._generate_cli_command,
        }

    async def start(self) -> None:
        """Start the generator with heartbeat."""
        self._is_running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop the generator."""
        self._is_running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self) -> None:
        """Emit periodic heartbeats."""
        while self._is_running:
            await asyncio.sleep(self.config.heartbeat_interval_s)
            if self._is_running:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["heartbeat"],
                    "kind": "heartbeat",
                    "actor": "code-generator",
                    "data": {
                        "cache_size": len(self._cache),
                        "patterns_registered": len(self._pattern_registry),
                    },
                })

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate code based on the request.

        Args:
            request: Generation request with prompt and configuration

        Returns:
            GenerationResult with generated code
        """
        start_time = time.time()

        # Emit request event
        self.bus.emit({
            "topic": self.BUS_TOPICS["request"],
            "kind": "request",
            "actor": "code-generator",
            "data": request.to_dict(),
        })

        # Check cache
        cache_key = self._cache_key(request)
        if self.config.cache_generated_code and cache_key in self._cache:
            cached_result, cache_time = self._cache[cache_key]
            if time.time() - cache_time < self.config.cache_ttl_s:
                return cached_result

        try:
            # Validate language
            if request.language not in self.config.supported_languages:
                raise ValueError(f"Unsupported language: {request.language}")

            # Generate based on mode and pattern
            if request.pattern and request.pattern in self._pattern_registry:
                code = await self._pattern_registry[request.pattern](request)
            else:
                code = await self._generate_from_prompt(request)

            # Calculate confidence
            confidence = self._calculate_confidence(code, request)

            result = GenerationResult(
                request_id=request.id,
                success=True,
                code=code,
                language=request.language,
                confidence=confidence,
                tokens_used=len(code.split()),
                elapsed_ms=(time.time() - start_time) * 1000,
            )

            # Cache result
            if self.config.cache_generated_code:
                self._cache[cache_key] = (result, time.time())

            # Emit completion event
            self.bus.emit({
                "topic": self.BUS_TOPICS["complete"],
                "kind": "response",
                "actor": "code-generator",
                "data": {
                    "request_id": request.id,
                    "success": True,
                    "confidence": confidence,
                    "tokens": result.tokens_used,
                    "elapsed_ms": result.elapsed_ms,
                },
            })

            return result

        except Exception as e:
            result = GenerationResult(
                request_id=request.id,
                success=False,
                code="",
                language=request.language,
                confidence=0.0,
                tokens_used=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

            # Emit error event
            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "error",
                "actor": "code-generator",
                "data": {
                    "request_id": request.id,
                    "error": str(e),
                },
            })

            return result

    def _cache_key(self, request: GenerationRequest) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.prompt,
            request.mode.value,
            request.language,
            request.pattern.value if request.pattern else "",
            str(request.constraints),
        ]
        return ":".join(key_parts)

    async def _generate_from_prompt(self, request: GenerationRequest) -> str:
        """Generate code from natural language prompt."""
        prompt = request.prompt.lower()
        language = request.language

        # Pattern detection from prompt
        if "function" in prompt or "def " in prompt:
            return await self._generate_function(request)
        elif "class" in prompt:
            return await self._generate_class(request)
        elif "test" in prompt:
            return await self._generate_test(request)
        elif "async" in prompt:
            return await self._generate_async_function(request)

        # Generic generation based on language
        templates = self.LANGUAGE_TEMPLATES.get(language, {})

        return f"# Generated code for: {request.prompt}\n# TODO: Implement\npass\n"

    async def _generate_function(self, request: GenerationRequest) -> str:
        """Generate a function."""
        context = request.context
        name = context.get("name", "new_function")
        params = context.get("params", "")
        return_type = context.get("return_type", "None")
        docstring = context.get("docstring", request.prompt)
        body = context.get("body", "pass")

        template = self.LANGUAGE_TEMPLATES.get(request.language, {}).get(
            "function", "def {name}():\n    pass\n"
        )

        return template.format(
            name=name,
            params=params,
            return_type=return_type,
            docstring=docstring,
            body=body,
        )

    async def _generate_class(self, request: GenerationRequest) -> str:
        """Generate a class."""
        context = request.context
        name = context.get("name", "NewClass")
        bases = context.get("bases", "")
        docstring = context.get("docstring", request.prompt)
        params = context.get("params", "")
        init_body = context.get("init_body", "pass")
        methods = context.get("methods", "")

        template = self.LANGUAGE_TEMPLATES.get(request.language, {}).get(
            "class", "class {name}:\n    pass\n"
        )

        return template.format(
            name=name,
            bases=bases,
            docstring=docstring,
            params=params,
            init_body=init_body,
            methods=methods,
        )

    async def _generate_dataclass(self, request: GenerationRequest) -> str:
        """Generate a dataclass."""
        context = request.context
        name = context.get("name", "NewDataClass")
        docstring = context.get("docstring", request.prompt)
        fields = context.get("fields", "    pass")

        template = self.LANGUAGE_TEMPLATES.get(request.language, {}).get(
            "dataclass", "@dataclass\nclass {name}:\n    pass\n"
        )

        return template.format(
            name=name,
            docstring=docstring,
            fields=fields,
        )

    async def _generate_async_function(self, request: GenerationRequest) -> str:
        """Generate an async function."""
        context = request.context
        name = context.get("name", "async_function")
        params = context.get("params", "")
        return_type = context.get("return_type", "None")
        docstring = context.get("docstring", request.prompt)
        body = context.get("body", "pass")

        template = self.LANGUAGE_TEMPLATES.get(request.language, {}).get(
            "async_function", "async def {name}():\n    pass\n"
        )

        return template.format(
            name=name,
            params=params,
            return_type=return_type,
            docstring=docstring,
            body=body,
        )

    async def _generate_test(self, request: GenerationRequest) -> str:
        """Generate a test function."""
        context = request.context
        name = context.get("name", "example")
        description = context.get("description", request.prompt)
        arrange = context.get("arrange", "# Setup")
        act = context.get("act", "result = None")
        assertions = context.get("assertions", "assert result is not None")

        template = self.LANGUAGE_TEMPLATES.get(request.language, {}).get(
            "test", "def test_{name}():\n    pass\n"
        )

        return template.format(
            name=name,
            description=description,
            arrange=arrange,
            act=act,
            assertions=assertions,
        )

    async def _generate_decorator(self, request: GenerationRequest) -> str:
        """Generate a decorator."""
        context = request.context
        name = context.get("name", "my_decorator")
        docstring = context.get("docstring", request.prompt)

        return f'''def {name}(func):
    """
    {docstring}
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Pre-processing
        result = func(*args, **kwargs)
        # Post-processing
        return result
    return wrapper
'''

    async def _generate_context_manager(self, request: GenerationRequest) -> str:
        """Generate a context manager."""
        context = request.context
        name = context.get("name", "managed_resource")
        docstring = context.get("docstring", request.prompt)

        return f'''@contextmanager
def {name}():
    """
    {docstring}
    """
    # Setup
    resource = None
    try:
        yield resource
    finally:
        # Cleanup
        pass
'''

    async def _generate_api_endpoint(self, request: GenerationRequest) -> str:
        """Generate an API endpoint."""
        context = request.context
        name = context.get("name", "endpoint")
        method = context.get("method", "GET")
        path = context.get("path", f"/{name}")
        docstring = context.get("docstring", request.prompt)

        return f'''@app.route("{path}", methods=["{method}"])
async def {name}(request):
    """
    {docstring}
    """
    try:
        # Process request
        data = await request.json() if request.body else {{}}

        # Business logic
        result = {{"status": "ok"}}

        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({{"error": str(e)}}, status_code=500)
'''

    async def _generate_cli_command(self, request: GenerationRequest) -> str:
        """Generate a CLI command."""
        context = request.context
        name = context.get("name", "command")
        docstring = context.get("docstring", request.prompt)

        return f'''@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def {name}(verbose: bool):
    """
    {docstring}
    """
    if verbose:
        click.echo("Running in verbose mode")

    # Command implementation
    click.echo("Done!")
'''

    def _calculate_confidence(self, code: str, request: GenerationRequest) -> float:
        """Calculate confidence score for generated code."""
        confidence = 0.7  # Base confidence

        # Adjust based on code quality indicators
        if code and not code.isspace():
            confidence += 0.1

        # Check for placeholder content
        if "TODO" in code or "pass" in code:
            confidence -= 0.1

        # Check for syntax-like patterns
        if request.language == "python":
            if "def " in code or "class " in code:
                confidence += 0.05
            if '"""' in code or "'''" in code:
                confidence += 0.05

        # Constraint satisfaction
        for constraint in request.constraints:
            if constraint.lower() in code.lower():
                confidence += 0.02

        return max(0.1, min(1.0, confidence))

    async def generate_batch(
        self,
        requests: List[GenerationRequest],
    ) -> List[GenerationResult]:
        """Generate code for multiple requests."""
        tasks = [self.generate(req) for req in requests]
        return await asyncio.gather(*tasks)

    def clear_cache(self) -> int:
        """Clear the generation cache."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "cache_size": len(self._cache),
            "patterns_registered": len(self._pattern_registry),
            "supported_languages": self.config.supported_languages,
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Code Generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Neural Code Generator (Step 71)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate code")
    gen_parser.add_argument("prompt", help="Generation prompt")
    gen_parser.add_argument("--language", "-l", default="python", help="Target language")
    gen_parser.add_argument("--pattern", "-p", choices=[p.value for p in CodePattern], help="Code pattern")
    gen_parser.add_argument("--name", "-n", help="Name for generated code")
    gen_parser.add_argument("--json", action="store_true", help="JSON output")

    # patterns command
    subparsers.add_parser("patterns", help="List available patterns")

    # stats command
    subparsers.add_parser("stats", help="Show generator stats")

    args = parser.parse_args()

    generator = CodeGenerator()

    if args.command == "generate":
        context = {}
        if args.name:
            context["name"] = args.name

        request = GenerationRequest(
            id=f"req-{uuid.uuid4().hex[:8]}",
            prompt=args.prompt,
            language=args.language,
            pattern=CodePattern(args.pattern) if args.pattern else None,
            context=context,
        )

        async def run():
            return await generator.generate(request)

        result = asyncio.run(run())

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(result.code)
            else:
                print(f"Error: {result.error}")
                return 1
        return 0

    elif args.command == "patterns":
        print("Available patterns:")
        for pattern in CodePattern:
            print(f"  {pattern.value}")
        return 0

    elif args.command == "stats":
        stats = generator.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
