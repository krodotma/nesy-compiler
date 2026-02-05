#!/usr/bin/env python3
"""
code_cli.py - Complete CLI Interface for Code Agent (Step 80)

PBTSO Phase: All Phases

Provides:
- Unified CLI for all code operations
- Generate, format, lint, check, refactor
- Template and snippet management
- API server control
- Interactive mode support

Bus Topics:
- code.cli.command
- code.cli.complete
- code.cli.error

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CLIConfig:
    """Configuration for the CLI."""
    working_dir: str = "/pluribus"
    output_format: str = "text"  # text, json
    verbose: bool = False
    quiet: bool = False
    color: bool = True
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "working_dir": self.working_dir,
            "output_format": self.output_format,
            "verbose": self.verbose,
            "quiet": self.quiet,
            "color": self.color,
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
# Output Helpers
# =============================================================================

class OutputHelper:
    """Helper for formatted output."""

    def __init__(self, config: CLIConfig):
        self.config = config

    def print(self, message: str) -> None:
        """Print a message."""
        if not self.config.quiet:
            print(message)

    def print_json(self, data: Any) -> None:
        """Print data as JSON."""
        print(json.dumps(data, indent=2))

    def print_success(self, message: str) -> None:
        """Print success message."""
        if self.config.quiet:
            return
        if self.config.color:
            print(f"\033[32m{message}\033[0m")
        else:
            print(f"OK: {message}")

    def print_error(self, message: str) -> None:
        """Print error message."""
        if self.config.color:
            print(f"\033[31m{message}\033[0m", file=sys.stderr)
        else:
            print(f"ERROR: {message}", file=sys.stderr)

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        if self.config.quiet:
            return
        if self.config.color:
            print(f"\033[33m{message}\033[0m")
        else:
            print(f"WARNING: {message}")

    def print_info(self, message: str) -> None:
        """Print info message."""
        if self.config.quiet or not self.config.verbose:
            return
        if self.config.color:
            print(f"\033[34m{message}\033[0m")
        else:
            print(f"INFO: {message}")

    def print_table(self, headers: List[str], rows: List[List[str]]) -> None:
        """Print a table."""
        if self.config.output_format == "json":
            data = [dict(zip(headers, row)) for row in rows]
            self.print_json(data)
            return

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        self.print(header_line)
        self.print("-" * len(header_line))

        # Print rows
        for row in rows:
            row_line = " | ".join(str(c).ljust(w) for c, w in zip(row, widths))
            self.print(row_line)


# =============================================================================
# Code CLI
# =============================================================================

class CodeCLI:
    """
    Complete CLI interface for Code Agent.

    PBTSO Phase: All Phases

    Commands:
    - generate: Generate code from prompts
    - format: Format code files
    - lint: Lint code files
    - check: Type check code files
    - refactor: Refactor code
    - template: Template management
    - snippet: Snippet management
    - docs: Documentation generation
    - api: API server control
    - stats: Show statistics

    Usage:
        cli = CodeCLI(config)
        exit_code = cli.run(args)
    """

    BUS_TOPICS = {
        "command": "code.cli.command",
        "complete": "code.cli.complete",
        "error": "code.cli.error",
    }

    def __init__(
        self,
        config: Optional[CLIConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or CLIConfig()
        self.bus = bus or LockedAgentBus()
        self.output = OutputHelper(self.config)

        # Components (lazy loaded)
        self._generator = None
        self._formatter = None
        self._linter = None
        self._type_checker = None
        self._refactorer = None
        self._template_engine = None
        self._snippet_manager = None
        self._doc_generator = None

    def _load_generator(self):
        if not self._generator:
            from ..generator import CodeGenerator
            self._generator = CodeGenerator()
        return self._generator

    def _load_formatter(self):
        if not self._formatter:
            from ..formatter import CodeFormatter
            self._formatter = CodeFormatter()
        return self._formatter

    def _load_linter(self):
        if not self._linter:
            from ..linter import UnifiedLinter
            self._linter = UnifiedLinter()
        return self._linter

    def _load_type_checker(self):
        if not self._type_checker:
            from ..typechecker import TypeChecker
            self._type_checker = TypeChecker()
        return self._type_checker

    def _load_refactorer(self):
        if not self._refactorer:
            from ..refactor import RefactoringEngine
            self._refactorer = RefactoringEngine()
        return self._refactorer

    def _load_template_engine(self):
        if not self._template_engine:
            from ..template import TemplateEngine
            self._template_engine = TemplateEngine()
        return self._template_engine

    def _load_snippet_manager(self):
        if not self._snippet_manager:
            from ..snippet import SnippetManager
            self._snippet_manager = SnippetManager()
        return self._snippet_manager

    def _load_doc_generator(self):
        if not self._doc_generator:
            from ..docs import DocumentationGenerator
            self._doc_generator = DocumentationGenerator()
        return self._doc_generator

    def build_parser(self) -> argparse.ArgumentParser:
        """Build argument parser."""
        parser = argparse.ArgumentParser(
            prog="code",
            description="Code Agent CLI - Neural-Guided Code Operations (Steps 71-80)",
        )

        # Global options
        parser.add_argument("--json", action="store_true", help="Output as JSON")
        parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
        parser.add_argument("-q", "--quiet", action="store_true", help="Quiet output")
        parser.add_argument("--no-color", action="store_true", help="Disable color output")
        parser.add_argument("--working-dir", "-C", default=".", help="Working directory")

        subparsers = parser.add_subparsers(dest="command", required=True)

        # generate command
        gen_parser = subparsers.add_parser("generate", aliases=["gen"], help="Generate code")
        gen_parser.add_argument("prompt", help="Generation prompt")
        gen_parser.add_argument("--language", "-l", default="python", help="Target language")
        gen_parser.add_argument("--pattern", "-p", help="Code pattern")
        gen_parser.add_argument("--name", "-n", help="Name for generated code")
        gen_parser.add_argument("--output", "-o", help="Output file")

        # format command
        fmt_parser = subparsers.add_parser("format", aliases=["fmt"], help="Format code")
        fmt_parser.add_argument("paths", nargs="+", help="Files or directories to format")
        fmt_parser.add_argument("--check", action="store_true", help="Check only, don't modify")
        fmt_parser.add_argument("--diff", action="store_true", help="Show diff")

        # lint command
        lint_parser = subparsers.add_parser("lint", help="Lint code")
        lint_parser.add_argument("paths", nargs="+", help="Files or directories to lint")
        lint_parser.add_argument("--fix", action="store_true", help="Apply auto-fixes")

        # check command (type checking)
        check_parser = subparsers.add_parser("check", help="Type check code")
        check_parser.add_argument("paths", nargs="+", help="Files or directories to check")
        check_parser.add_argument("--strict", action="store_true", help="Strict mode")
        check_parser.add_argument("--suggest", action="store_true", help="Suggest type annotations")

        # refactor command
        refactor_parser = subparsers.add_parser("refactor", aliases=["ref"], help="Refactor code")
        refactor_subparsers = refactor_parser.add_subparsers(dest="refactor_type", required=True)

        rename_parser = refactor_subparsers.add_parser("rename", help="Rename symbol")
        rename_parser.add_argument("file", help="Target file")
        rename_parser.add_argument("old_name", help="Old name")
        rename_parser.add_argument("new_name", help="New name")
        rename_parser.add_argument("--scope", nargs="+", help="Files to update")

        extract_parser = refactor_subparsers.add_parser("extract", help="Extract method")
        extract_parser.add_argument("file", help="Target file")
        extract_parser.add_argument("--name", "-n", required=True, help="New method name")
        extract_parser.add_argument("--start", type=int, required=True, help="Start line")
        extract_parser.add_argument("--end", type=int, required=True, help="End line")

        refs_parser = refactor_subparsers.add_parser("find-refs", help="Find references")
        refs_parser.add_argument("file", help="File with symbol")
        refs_parser.add_argument("symbol", help="Symbol name")

        # template command
        template_parser = subparsers.add_parser("template", aliases=["tpl"], help="Template operations")
        template_subparsers = template_parser.add_subparsers(dest="template_action", required=True)

        tpl_list_parser = template_subparsers.add_parser("list", help="List templates")
        tpl_list_parser.add_argument("--language", "-l", help="Filter by language")
        tpl_list_parser.add_argument("--category", "-c", help="Filter by category")

        tpl_render_parser = template_subparsers.add_parser("render", help="Render template")
        tpl_render_parser.add_argument("template_id", help="Template ID")
        tpl_render_parser.add_argument("--var", "-v", nargs=2, action="append", metavar=("NAME", "VALUE"))
        tpl_render_parser.add_argument("--output", "-o", help="Output file")

        tpl_show_parser = template_subparsers.add_parser("show", help="Show template")
        tpl_show_parser.add_argument("template_id", help="Template ID")

        # snippet command
        snippet_parser = subparsers.add_parser("snippet", aliases=["snip"], help="Snippet operations")
        snippet_subparsers = snippet_parser.add_subparsers(dest="snippet_action", required=True)

        snip_search_parser = snippet_subparsers.add_parser("search", help="Search snippets")
        snip_search_parser.add_argument("query", help="Search query")
        snip_search_parser.add_argument("--language", "-l", help="Filter by language")
        snip_search_parser.add_argument("--limit", "-n", type=int, default=10)

        snip_add_parser = snippet_subparsers.add_parser("add", help="Add snippet")
        snip_add_parser.add_argument("name", help="Snippet name")
        snip_add_parser.add_argument("--code", "-c", help="Snippet code")
        snip_add_parser.add_argument("--language", "-l", default="python")
        snip_add_parser.add_argument("--tags", "-t", nargs="+")
        snip_add_parser.add_argument("--description", "-d")

        snip_get_parser = snippet_subparsers.add_parser("get", help="Get snippet")
        snip_get_parser.add_argument("snippet_id", help="Snippet ID")

        snip_use_parser = snippet_subparsers.add_parser("use", help="Use snippet")
        snip_use_parser.add_argument("snippet_id", help="Snippet ID")

        snippet_subparsers.add_parser("popular", help="Show popular snippets")

        # docs command
        docs_parser = subparsers.add_parser("docs", help="Documentation operations")
        docs_subparsers = docs_parser.add_subparsers(dest="docs_action", required=True)

        docs_gen_parser = docs_subparsers.add_parser("generate", help="Generate documentation")
        docs_gen_parser.add_argument("file", help="Python file")
        docs_gen_parser.add_argument("--format", "-f", choices=["markdown", "json", "rst"], default="markdown")
        docs_gen_parser.add_argument("--output", "-o", help="Output file")

        docs_validate_parser = docs_subparsers.add_parser("validate", help="Validate documentation")
        docs_validate_parser.add_argument("file", help="Python file")

        docs_docstring_parser = docs_subparsers.add_parser("docstring", help="Generate docstring")
        docs_docstring_parser.add_argument("--style", choices=["google", "numpy"], default="google")

        # api command
        api_parser = subparsers.add_parser("api", help="API server operations")
        api_subparsers = api_parser.add_subparsers(dest="api_action", required=True)

        api_serve_parser = api_subparsers.add_parser("serve", help="Start API server")
        api_serve_parser.add_argument("--host", default="0.0.0.0")
        api_serve_parser.add_argument("--port", type=int, default=8079)

        api_subparsers.add_parser("openapi", help="Print OpenAPI spec")

        # stats command
        subparsers.add_parser("stats", help="Show Code Agent statistics")

        # version command
        subparsers.add_parser("version", help="Show version")

        return parser

    def run(self, argv: Optional[List[str]] = None) -> int:
        """Run the CLI with given arguments."""
        parser = self.build_parser()
        args = parser.parse_args(argv)

        # Apply global options
        if args.json:
            self.config.output_format = "json"
        if args.verbose:
            self.config.verbose = True
        if args.quiet:
            self.config.quiet = True
        if args.no_color:
            self.config.color = False
        if args.working_dir:
            self.config.working_dir = args.working_dir
            os.chdir(args.working_dir)

        self.output = OutputHelper(self.config)

        # Emit command event
        self.bus.emit({
            "topic": self.BUS_TOPICS["command"],
            "kind": "cli",
            "actor": "code-cli",
            "data": {
                "command": args.command,
                "args": vars(args),
            },
        })

        try:
            # Dispatch to command handler
            if args.command in {"generate", "gen"}:
                return self._cmd_generate(args)
            elif args.command in {"format", "fmt"}:
                return self._cmd_format(args)
            elif args.command == "lint":
                return self._cmd_lint(args)
            elif args.command == "check":
                return self._cmd_check(args)
            elif args.command in {"refactor", "ref"}:
                return self._cmd_refactor(args)
            elif args.command in {"template", "tpl"}:
                return self._cmd_template(args)
            elif args.command in {"snippet", "snip"}:
                return self._cmd_snippet(args)
            elif args.command == "docs":
                return self._cmd_docs(args)
            elif args.command == "api":
                return self._cmd_api(args)
            elif args.command == "stats":
                return self._cmd_stats(args)
            elif args.command == "version":
                self.output.print("Code Agent CLI v0.1.0")
                self.output.print("Steps 71-80 of OAGENT 300-step plan")
                return 0

            return 1

        except Exception as e:
            self.output.print_error(str(e))
            self.bus.emit({
                "topic": self.BUS_TOPICS["error"],
                "kind": "error",
                "level": "error",
                "actor": "code-cli",
                "data": {"error": str(e)},
            })
            return 1

    # =========================================================================
    # Command Handlers
    # =========================================================================

    def _cmd_generate(self, args) -> int:
        """Handle generate command."""
        generator = self._load_generator()

        from ..generator import GenerationRequest, CodePattern

        context = {}
        if args.name:
            context["name"] = args.name

        request = GenerationRequest(
            id=f"cli-{uuid.uuid4().hex[:8]}",
            prompt=args.prompt,
            language=args.language,
            pattern=CodePattern(args.pattern) if args.pattern else None,
            context=context,
        )

        result = asyncio.run(generator.generate(request))

        if self.config.output_format == "json":
            self.output.print_json(result.to_dict())
        else:
            if result.success:
                if args.output:
                    Path(args.output).write_text(result.code)
                    self.output.print_success(f"Written to {args.output}")
                else:
                    print(result.code)
            else:
                self.output.print_error(result.error)
                return 1

        return 0

    def _cmd_format(self, args) -> int:
        """Handle format command."""
        formatter = self._load_formatter()

        all_results = []

        for path_str in args.paths:
            path = Path(path_str)

            if path.is_dir():
                result = formatter.format_directory(path, check_only=args.check)
                all_results.extend(result.results)
            else:
                result = formatter.format_file(path, check_only=args.check)
                all_results.append(result)

        if self.config.output_format == "json":
            self.output.print_json([r.to_dict() for r in all_results])
        else:
            modified = sum(1 for r in all_results if r.was_modified)
            failed = sum(1 for r in all_results if not r.success)

            self.output.print(f"Processed {len(all_results)} files")
            self.output.print(f"  Modified: {modified}")
            self.output.print(f"  Failed: {failed}")

            if args.check and modified > 0:
                self.output.print("\nFiles needing format:")
                for r in all_results:
                    if r.was_modified:
                        self.output.print(f"  {r.file_path}")
                return 1

        return 0

    def _cmd_lint(self, args) -> int:
        """Handle lint command."""
        linter = self._load_linter()

        all_issues = []
        total_errors = 0

        for path_str in args.paths:
            path = Path(path_str)

            if args.fix and path.is_file():
                result = linter.fix_file(path)
                self.output.print(f"{path}: Fixed {result.issues_fixed} issues")
            else:
                if path.is_dir():
                    result = linter.lint_directory(path)
                else:
                    result = linter.lint_file(path)

                all_issues.extend(result.issues)
                total_errors += result.errors

        if self.config.output_format == "json":
            self.output.print_json([i.to_dict() for i in all_issues])
        else:
            if all_issues:
                self.output.print(f"\nFound {len(all_issues)} issues ({total_errors} errors)")
                for issue in all_issues[:50]:
                    sev = issue.severity.value.upper()
                    self.output.print(f"  {issue.file_path}:{issue.line}:{issue.column} [{sev}] {issue.code}: {issue.message}")

        return 1 if total_errors > 0 else 0

    def _cmd_check(self, args) -> int:
        """Handle check command."""
        checker = self._load_type_checker()

        all_issues = []
        total_errors = 0

        for path_str in args.paths:
            path = Path(path_str)

            if args.suggest:
                suggestions = checker.suggest_types(path)
                if self.config.output_format == "json":
                    self.output.print_json([s.to_dict() for s in suggestions])
                else:
                    for s in suggestions:
                        self.output.print(f"  Line {s.line}: {s.symbol}: {s.suggested_type}")
            else:
                if path.is_dir():
                    result = checker.check_directory(path)
                else:
                    result = checker.check_file(path)

                all_issues.extend(result.issues)
                total_errors += result.errors

        if not args.suggest:
            if self.config.output_format == "json":
                self.output.print_json([i.to_dict() for i in all_issues])
            else:
                if all_issues:
                    self.output.print(f"\nFound {len(all_issues)} type issues ({total_errors} errors)")
                    for issue in all_issues[:50]:
                        code_str = f"[{issue.code}] " if issue.code else ""
                        self.output.print(f"  {issue.file_path}:{issue.line}:{issue.column} {code_str}{issue.message}")

        return 1 if total_errors > 0 else 0

    def _cmd_refactor(self, args) -> int:
        """Handle refactor command."""
        refactorer = self._load_refactorer()

        from ..refactor import RefactoringOperation, RefactoringType

        if args.refactor_type == "rename":
            operation = RefactoringOperation(
                id=f"cli-{uuid.uuid4().hex[:8]}",
                refactor_type=RefactoringType.RENAME,
                target=args.file,
                options={"old_name": args.old_name, "new_name": args.new_name},
                scope=args.scope or [],
            )

            result = refactorer.refactor(operation)

            if self.config.output_format == "json":
                self.output.print_json(result.to_dict())
            else:
                if result.success:
                    self.output.print_success(f"Renamed {args.old_name} to {args.new_name}")
                    self.output.print(f"Modified {result.files_modified} files")
                else:
                    self.output.print_error(result.error)
                    return 1

        elif args.refactor_type == "extract":
            operation = RefactoringOperation(
                id=f"cli-{uuid.uuid4().hex[:8]}",
                refactor_type=RefactoringType.EXTRACT_METHOD,
                target=args.file,
                options={
                    "method_name": args.name,
                    "start_line": args.start,
                    "end_line": args.end,
                },
            )

            result = refactorer.refactor(operation)

            if self.config.output_format == "json":
                self.output.print_json(result.to_dict())
            else:
                if result.success:
                    self.output.print_success(f"Extracted method: {args.name}")
                else:
                    self.output.print_error(result.error)
                    return 1

        elif args.refactor_type == "find-refs":
            refs = refactorer.find_references(args.symbol, args.file)

            if self.config.output_format == "json":
                self.output.print_json([r.to_dict() for r in refs])
            else:
                self.output.print(f"Found {len(refs)} references to {args.symbol}:")
                for ref in refs:
                    marker = " (definition)" if ref.is_definition else ""
                    self.output.print(f"  {ref.file_path}:{ref.line}:{ref.column}{marker}")

        return 0

    def _cmd_template(self, args) -> int:
        """Handle template command."""
        engine = self._load_template_engine()

        if args.template_action == "list":
            templates = engine.list_templates(
                language=args.language,
                category=args.category,
            )

            if self.config.output_format == "json":
                self.output.print_json([t.to_dict() for t in templates])
            else:
                self.output.print_table(
                    ["ID", "Name", "Language", "Category"],
                    [[t.id, t.name, t.language, t.category] for t in templates]
                )

        elif args.template_action == "render":
            variables = {}
            if args.var:
                for name, value in args.var:
                    try:
                        variables[name] = json.loads(value)
                    except json.JSONDecodeError:
                        variables[name] = value

            result = engine.render(args.template_id, variables)

            if result.success:
                if args.output:
                    Path(args.output).write_text(result.content)
                    self.output.print_success(f"Written to {args.output}")
                else:
                    print(result.content)
            else:
                self.output.print_error(result.error)
                return 1

        elif args.template_action == "show":
            template = engine.get_template(args.template_id)

            if not template:
                self.output.print_error(f"Template not found: {args.template_id}")
                return 1

            if self.config.output_format == "json":
                self.output.print_json(template.to_dict())
            else:
                self.output.print(f"ID: {template.id}")
                self.output.print(f"Name: {template.name}")
                self.output.print(f"Language: {template.language}")
                self.output.print(f"Variables: {', '.join(v.name for v in template.variables)}")
                self.output.print(f"\nContent:\n{template.content}")

        return 0

    def _cmd_snippet(self, args) -> int:
        """Handle snippet command."""
        manager = self._load_snippet_manager()

        if args.snippet_action == "search":
            results = manager.search(
                args.query,
                language=args.language,
                limit=args.limit,
            )

            if self.config.output_format == "json":
                self.output.print_json([r.to_dict() for r in results])
            else:
                for r in results:
                    self.output.print(f"{r.snippet.id}: {r.snippet.name} ({r.score:.2f})")
                    if r.highlights:
                        self.output.print(f"  {', '.join(r.highlights)}")

        elif args.snippet_action == "add":
            from ..snippet import CodeSnippet, SnippetCategory

            code = args.code
            if code is None:
                code = sys.stdin.read()

            snippet = CodeSnippet(
                id=f"snippet-{uuid.uuid4().hex[:8]}",
                name=args.name,
                code=code,
                language=args.language,
                tags=args.tags or [],
                description=args.description or "",
            )

            snippet_id = manager.add(snippet)
            self.output.print_success(f"Added snippet: {snippet_id}")

        elif args.snippet_action == "get":
            snippet = manager.get(args.snippet_id)

            if not snippet:
                self.output.print_error(f"Snippet not found: {args.snippet_id}")
                return 1

            if self.config.output_format == "json":
                self.output.print_json(snippet.to_dict())
            else:
                self.output.print(f"ID: {snippet.id}")
                self.output.print(f"Name: {snippet.name}")
                self.output.print(f"Language: {snippet.language}")
                self.output.print(f"Tags: {', '.join(snippet.tags)}")
                self.output.print(f"\n{snippet.code}")

        elif args.snippet_action == "use":
            code = manager.use(args.snippet_id)

            if code is None:
                self.output.print_error(f"Snippet not found: {args.snippet_id}")
                return 1

            print(code)

        elif args.snippet_action == "popular":
            snippets = manager.get_popular(10)

            if self.config.output_format == "json":
                self.output.print_json([s.to_dict() for s in snippets])
            else:
                for s in snippets:
                    self.output.print(f"{s.id}: {s.name} ({s.usage_count} uses)")

        return 0

    def _cmd_docs(self, args) -> int:
        """Handle docs command."""
        generator = self._load_doc_generator()

        from ..docs import DocFormat

        if args.docs_action == "generate":
            format_map = {
                "markdown": DocFormat.MARKDOWN,
                "json": DocFormat.JSON,
                "rst": DocFormat.RST,
            }

            result = generator.generate(Path(args.file), format_map[args.format])

            if result.success:
                if args.output:
                    Path(args.output).write_text(result.content)
                    self.output.print_success(f"Written to {args.output}")
                else:
                    print(result.content)
            else:
                self.output.print_error(result.error)
                return 1

        elif args.docs_action == "validate":
            result = generator.validate(Path(args.file))

            if self.config.output_format == "json":
                self.output.print_json(result)
            else:
                self.output.print(f"Coverage: {result['coverage_percent']}%")
                self.output.print(f"Documented: {result['documented_items']}/{result['total_items']}")
                if result['missing_docstrings']:
                    self.output.print("\nMissing docstrings:")
                    for item in result['missing_docstrings']:
                        self.output.print(f"  - {item}")

        elif args.docs_action == "docstring":
            from ..docs import DocstringStyle

            code = sys.stdin.read()
            style_map = {
                "google": DocstringStyle.GOOGLE,
                "numpy": DocstringStyle.NUMPY,
            }
            docstring = generator.generate_docstring(code, style_map[args.style])
            print(docstring)

        return 0

    def _cmd_api(self, args) -> int:
        """Handle api command."""
        from ..api import CodeAPI, APIConfig

        if args.api_action == "serve":
            config = APIConfig(
                host=args.host,
                port=args.port,
            )
            api = CodeAPI(config)

            self.output.print(f"Code API would serve on http://{config.host}:{config.port}")
            self.output.print("Note: Actual HTTP server requires aiohttp/fastapi dependency")
            self.output.print("\nAvailable endpoints:")
            for method, routes in api._routes.items():
                for path in routes:
                    self.output.print(f"  {method} {path}")

        elif args.api_action == "openapi":
            config = APIConfig()
            api = CodeAPI(config)
            spec = api.get_openapi_spec()

            if self.config.output_format == "json":
                self.output.print_json(spec)
            else:
                print(json.dumps(spec, indent=2))

        return 0

    def _cmd_stats(self, args) -> int:
        """Handle stats command."""
        stats = {
            "components": {
                "generator": "available",
                "formatter": "available",
                "linter": "available",
                "type_checker": "available",
                "refactorer": "available",
                "template_engine": "available",
                "snippet_manager": "available",
                "doc_generator": "available",
            },
            "steps": "71-80",
            "protocol": "DKIN v30, CITIZEN v2",
            "config": self.config.to_dict(),
        }

        if self.config.output_format == "json":
            self.output.print_json(stats)
        else:
            self.output.print("Code Agent Statistics")
            self.output.print(f"  Steps: {stats['steps']}")
            self.output.print(f"  Protocol: {stats['protocol']}")
            self.output.print("\nComponents:")
            for comp, status in stats['components'].items():
                self.output.print(f"  {comp}: {status}")

        return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    cli = CodeCLI()
    return cli.run(argv)


if __name__ == "__main__":
    sys.exit(main())
