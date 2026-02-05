#!/usr/bin/env python3
"""
research_cli.py - Research CLI (Step 30)

Complete command-line interface for research operations.
Provides unified access to all research functionality.

PBTSO Phase: INTERFACE

Bus Topics:
- a2a.research.cli.execute
- research.cli.complete

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class CLIConfig:
    """Configuration for Research CLI."""

    output_format: str = "text"  # text, json, markdown
    color_enabled: bool = True
    verbose: bool = False
    project_root: Optional[str] = None
    config_path: Optional[str] = None
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = os.environ.get("PLURIBUS_ROOT", os.getcwd())
        if self.bus_path is None:
            self.bus_path = f"{self.project_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Color Output
# ============================================================================


class Colors:
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    @classmethod
    def disable(cls):
        """Disable colors."""
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""


# ============================================================================
# Research CLI
# ============================================================================


class ResearchCLI:
    """
    Complete CLI for research operations.

    Commands:
    - search: Search the codebase
    - find: Find symbol definitions
    - usages: Find symbol usages
    - explain: Explain code
    - deps: Show dependencies
    - context: Assemble context
    - index: Manage indexes
    - repo: Manage repositories
    - stats: Show statistics
    - serve: Start API server
    - watch: Watch for changes

    PBTSO Phase: INTERFACE

    Example:
        cli = ResearchCLI()
        cli.run(["search", "UserService"])
    """

    VERSION = "0.1.0"

    def __init__(
        self,
        config: Optional[CLIConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the Research CLI.

        Args:
            config: CLI configuration
            bus: AgentBus for event emission
        """
        self.config = config or CLIConfig()
        self.bus = bus or AgentBus()

        if not self.config.color_enabled or not sys.stdout.isatty():
            Colors.disable()

    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI with given arguments.

        Args:
            args: Command line arguments (uses sys.argv if None)

        Returns:
            Exit code
        """
        parser = self._create_parser()
        parsed = parser.parse_args(args)

        # Handle global options
        if parsed.json:
            self.config.output_format = "json"
        if parsed.verbose:
            self.config.verbose = True
        if parsed.no_color:
            Colors.disable()

        # Execute command
        if hasattr(parsed, "func"):
            try:
                return parsed.func(parsed)
            except KeyboardInterrupt:
                self._print_error("Interrupted")
                return 130
            except Exception as e:
                self._print_error(str(e))
                if self.config.verbose:
                    import traceback
                    traceback.print_exc()
                return 1
        else:
            parser.print_help()
            return 0

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="research",
            description="Research Agent CLI - Explore and understand codebases",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  research search "authentication handler"
  research find UserService
  research usages "AuthManager" --json
  research deps src/main.py
  research context src/*.py --max-tokens 10000
  research serve --port 8080
            """,
        )

        # Global options
        parser.add_argument(
            "--version", "-V",
            action="version",
            version=f"Research Agent CLI v{self.VERSION}",
        )
        parser.add_argument(
            "--json", "-j",
            action="store_true",
            help="Output as JSON",
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Verbose output",
        )
        parser.add_argument(
            "--no-color",
            action="store_true",
            help="Disable colored output",
        )
        parser.add_argument(
            "--root",
            help="Project root directory",
        )

        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Commands")

        # search command
        search_parser = subparsers.add_parser(
            "search", aliases=["s"],
            help="Search the codebase",
        )
        search_parser.add_argument("query", help="Search query")
        search_parser.add_argument("--max", "-n", type=int, default=20, help="Maximum results")
        search_parser.add_argument("--path", "-p", help="Limit to path")
        search_parser.add_argument("--type", "-t", help="Filter by type (class, function, etc.)")
        search_parser.set_defaults(func=self._cmd_search)

        # find command
        find_parser = subparsers.add_parser(
            "find", aliases=["f"],
            help="Find symbol definition",
        )
        find_parser.add_argument("symbol", help="Symbol name")
        find_parser.add_argument("--kind", "-k", help="Symbol kind (class, function)")
        find_parser.set_defaults(func=self._cmd_find)

        # usages command
        usages_parser = subparsers.add_parser(
            "usages", aliases=["u"],
            help="Find symbol usages",
        )
        usages_parser.add_argument("symbol", help="Symbol name")
        usages_parser.add_argument("--file", "-f", help="Starting file")
        usages_parser.set_defaults(func=self._cmd_usages)

        # explain command
        explain_parser = subparsers.add_parser(
            "explain", aliases=["e"],
            help="Explain code",
        )
        explain_parser.add_argument("path", help="File path")
        explain_parser.add_argument("--symbol", "-s", help="Specific symbol")
        explain_parser.add_argument("--line", "-l", type=int, help="Line number")
        explain_parser.set_defaults(func=self._cmd_explain)

        # deps command
        deps_parser = subparsers.add_parser(
            "deps", aliases=["d"],
            help="Show dependencies",
        )
        deps_parser.add_argument("path", help="File path")
        deps_parser.add_argument("--depth", type=int, default=3, help="Maximum depth")
        deps_parser.add_argument("--reverse", "-r", action="store_true", help="Show dependents")
        deps_parser.set_defaults(func=self._cmd_deps)

        # context command
        context_parser = subparsers.add_parser(
            "context", aliases=["ctx"],
            help="Assemble context",
        )
        context_parser.add_argument("files", nargs="+", help="Files to include")
        context_parser.add_argument("--max-tokens", "-t", type=int, default=50000, help="Max tokens")
        context_parser.add_argument("--format", choices=["markdown", "xml", "plain"], default="markdown")
        context_parser.set_defaults(func=self._cmd_context)

        # index command
        index_parser = subparsers.add_parser(
            "index", aliases=["i"],
            help="Manage indexes",
        )
        index_sub = index_parser.add_subparsers(dest="index_cmd")

        index_build = index_sub.add_parser("build", help="Build/rebuild index")
        index_build.add_argument("--path", default=".", help="Path to index")
        index_build.set_defaults(func=self._cmd_index_build)

        index_stats = index_sub.add_parser("stats", help="Show index statistics")
        index_stats.set_defaults(func=self._cmd_index_stats)

        index_clear = index_sub.add_parser("clear", help="Clear index")
        index_clear.set_defaults(func=self._cmd_index_clear)

        # repo command
        repo_parser = subparsers.add_parser(
            "repo", aliases=["r"],
            help="Manage repositories",
        )
        repo_sub = repo_parser.add_subparsers(dest="repo_cmd")

        repo_add = repo_sub.add_parser("add", help="Add repository")
        repo_add.add_argument("path", help="Repository path")
        repo_add.add_argument("--name", help="Display name")
        repo_add.set_defaults(func=self._cmd_repo_add)

        repo_list = repo_sub.add_parser("list", help="List repositories")
        repo_list.set_defaults(func=self._cmd_repo_list)

        repo_remove = repo_sub.add_parser("remove", help="Remove repository")
        repo_remove.add_argument("repo_id", help="Repository ID")
        repo_remove.set_defaults(func=self._cmd_repo_remove)

        # stats command
        stats_parser = subparsers.add_parser(
            "stats",
            help="Show statistics",
        )
        stats_parser.set_defaults(func=self._cmd_stats)

        # serve command
        serve_parser = subparsers.add_parser(
            "serve",
            help="Start API server",
        )
        serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
        serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind")
        serve_parser.add_argument("--api-key", help="API key for auth")
        serve_parser.set_defaults(func=self._cmd_serve)

        # watch command
        watch_parser = subparsers.add_parser(
            "watch", aliases=["w"],
            help="Watch for changes",
        )
        watch_parser.add_argument("--path", default=".", help="Path to watch")
        watch_parser.set_defaults(func=self._cmd_watch)

        return parser

    # ========================================================================
    # Command Handlers
    # ========================================================================

    def _cmd_search(self, args) -> int:
        """Handle search command."""
        self._emit_cli_event("search", {"query": args.query})

        self._print_header(f"Searching: {args.query}")

        # Import orchestrator
        try:
            from ..orchestrator import ResearchOrchestrator, ResearchOrchestratorConfig

            async def run():
                config = ResearchOrchestratorConfig(root=self.config.project_root)
                orchestrator = ResearchOrchestrator(config)
                await orchestrator.initialize()

                result = await orchestrator.research(
                    args.query,
                    max_results=args.max,
                )

                return result

            result = asyncio.run(run())

            if self.config.output_format == "json":
                output = {
                    "query": result.query.query,
                    "intent": result.intent.value,
                    "results": result.results[:args.max],
                    "execution_time_ms": result.execution_time_ms,
                }
                print(json.dumps(output, indent=2))
            else:
                self._print_results(result.results[:args.max])
                self._print_footer(f"Found {len(result.results)} results in {result.execution_time_ms:.1f}ms")

        except ImportError as e:
            self._print_error(f"Module not available: {e}")
            return 1

        return 0

    def _cmd_find(self, args) -> int:
        """Handle find command."""
        self._emit_cli_event("find", {"symbol": args.symbol})

        self._print_header(f"Finding: {args.symbol}")

        try:
            from ..index.symbol_store import SymbolIndexStore

            store = SymbolIndexStore()
            symbols = store.query(name=args.symbol, kind=args.kind, limit=20)

            if self.config.output_format == "json":
                print(json.dumps([s.to_dict() for s in symbols], indent=2))
            else:
                if not symbols:
                    self._print_warning(f"No definition found for '{args.symbol}'")
                else:
                    for s in symbols:
                        self._print_symbol(s)

            store.close()

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_usages(self, args) -> int:
        """Handle usages command."""
        self._emit_cli_event("usages", {"symbol": args.symbol})

        self._print_header(f"Finding usages: {args.symbol}")

        try:
            from ..analysis.reference_resolver import ReferenceResolver

            resolver = ReferenceResolver(root=Path(self.config.project_root))
            usages = resolver.find_usages(args.symbol)

            if self.config.output_format == "json":
                print(json.dumps([u.to_dict() for u in usages], indent=2))
            else:
                if not usages:
                    self._print_warning(f"No usages found for '{args.symbol}'")
                else:
                    self._print_success(f"Found {len(usages)} usages:")
                    for u in usages[:20]:
                        path = u.reference.source_path
                        line = u.reference.source_line
                        ctx = u.reference.context or ""
                        print(f"  {Colors.CYAN}{path}{Colors.RESET}:{line}")
                        if ctx:
                            print(f"    {Colors.DIM}{ctx.strip()[:80]}{Colors.RESET}")

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_explain(self, args) -> int:
        """Handle explain command."""
        self._emit_cli_event("explain", {"path": args.path})

        path = Path(args.path)
        if not path.exists():
            self._print_error(f"File not found: {args.path}")
            return 1

        self._print_header(f"Explaining: {args.path}")

        # Read file
        content = path.read_text(errors="ignore")
        lines = content.split("\n")

        # Show basic info
        print(f"\n{Colors.BOLD}File:{Colors.RESET} {path}")
        print(f"{Colors.BOLD}Lines:{Colors.RESET} {len(lines)}")
        print(f"{Colors.BOLD}Language:{Colors.RESET} {path.suffix}")

        if args.line:
            # Show specific line with context
            start = max(0, args.line - 5)
            end = min(len(lines), args.line + 5)
            print(f"\n{Colors.BOLD}Context (lines {start+1}-{end}):{Colors.RESET}")
            for i in range(start, end):
                marker = "->" if i == args.line - 1 else "  "
                print(f"  {marker} {i+1:4} | {lines[i]}")

        return 0

    def _cmd_deps(self, args) -> int:
        """Handle deps command."""
        self._emit_cli_event("deps", {"path": args.path})

        self._print_header(f"Dependencies: {args.path}")

        try:
            from ..graph.dependency_builder import DependencyGraphBuilder

            builder = DependencyGraphBuilder(root=Path(self.config.project_root))

            if args.reverse:
                deps = builder.get_dependents(args.path)
                label = "Imported by"
            else:
                deps = builder.get_dependencies(args.path)
                label = "Imports"

            if self.config.output_format == "json":
                print(json.dumps({"path": args.path, "direction": label.lower(), "dependencies": list(deps)}, indent=2))
            else:
                print(f"\n{Colors.BOLD}{label}:{Colors.RESET}")
                if not deps:
                    self._print_warning("  No dependencies found")
                else:
                    for dep in sorted(deps)[:30]:
                        print(f"  {Colors.CYAN}{dep}{Colors.RESET}")

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_context(self, args) -> int:
        """Handle context command."""
        self._emit_cli_event("context", {"files": args.files})

        try:
            from ..search.context_assembler import ContextAssembler, ContextPriority

            assembler = ContextAssembler()

            for i, path in enumerate(args.files):
                priority = ContextPriority.HIGH if i == 0 else ContextPriority.MEDIUM
                assembler.add_file(path, priority=priority)

            context = assembler.assemble(
                max_tokens=args.max_tokens,
                format_style=args.format,
            )

            if self.config.output_format == "json":
                print(json.dumps(context.to_dict(), indent=2))
            else:
                print(context.formatted_content)
                print(f"\n{Colors.DIM}--- Tokens: {context.total_tokens} | Sources: {len(context.sources)} ---{Colors.RESET}", file=sys.stderr)

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_index_build(self, args) -> int:
        """Handle index build command."""
        self._emit_cli_event("index.build", {"path": args.path})

        self._print_header("Building index...")

        try:
            from ..scanner import CodebaseScanner

            scanner = CodebaseScanner()

            async def run():
                result = await scanner.scan(args.path)
                return result

            result = asyncio.run(run())

            self._print_success(f"Indexed {result.files_scanned} files, {result.symbols_found} symbols")

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_index_stats(self, args) -> int:
        """Handle index stats command."""
        self._emit_cli_event("index.stats", {})

        try:
            from ..index.symbol_store import SymbolIndexStore

            store = SymbolIndexStore()
            stats = store.stats()
            store.close()

            if self.config.output_format == "json":
                print(json.dumps(stats, indent=2))
            else:
                self._print_header("Index Statistics")
                print(f"  Total Symbols: {stats['total']}")
                print(f"  Files: {stats['file_count']}")
                print(f"\n  By Kind:")
                for kind, count in sorted(stats['by_kind'].items()):
                    print(f"    {kind}: {count}")
                print(f"\n  By Language:")
                for lang, count in sorted(stats['by_language'].items()):
                    print(f"    {lang}: {count}")

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_index_clear(self, args) -> int:
        """Handle index clear command."""
        self._emit_cli_event("index.clear", {})

        try:
            from ..index.symbol_store import SymbolIndexStore

            store = SymbolIndexStore()
            store.clear()
            store.close()

            self._print_success("Index cleared")

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_repo_add(self, args) -> int:
        """Handle repo add command."""
        self._emit_cli_event("repo.add", {"path": args.path})

        try:
            from ..multi_repo.multi_repo_manager import MultiRepoManager

            manager = MultiRepoManager()
            repo = manager.register_repo(args.path, name=args.name)

            if self.config.output_format == "json":
                print(json.dumps(repo.to_dict(), indent=2))
            else:
                self._print_success(f"Added repository: {repo.name} ({repo.id})")

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_repo_list(self, args) -> int:
        """Handle repo list command."""
        self._emit_cli_event("repo.list", {})

        try:
            from ..multi_repo.multi_repo_manager import MultiRepoManager

            manager = MultiRepoManager()
            repos = manager.list_repos()

            if self.config.output_format == "json":
                print(json.dumps([r.to_dict() for r in repos], indent=2))
            else:
                self._print_header(f"Repositories ({len(repos)})")
                for r in repos:
                    status_color = Colors.GREEN if r.status.value == "active" else Colors.YELLOW
                    print(f"  [{r.id[:8]}] {Colors.BOLD}{r.name}{Colors.RESET}")
                    print(f"    Path: {r.path}")
                    print(f"    Status: {status_color}{r.status.value}{Colors.RESET}")

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_repo_remove(self, args) -> int:
        """Handle repo remove command."""
        self._emit_cli_event("repo.remove", {"repo_id": args.repo_id})

        try:
            from ..multi_repo.multi_repo_manager import MultiRepoManager

            manager = MultiRepoManager()
            if manager.unregister_repo(args.repo_id):
                self._print_success(f"Removed repository: {args.repo_id}")
            else:
                self._print_error(f"Repository not found: {args.repo_id}")
                return 1

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_stats(self, args) -> int:
        """Handle stats command."""
        self._emit_cli_event("stats", {})

        stats = {
            "version": self.VERSION,
            "project_root": self.config.project_root,
            "timestamp": time.time(),
        }

        # Try to get index stats
        try:
            from ..index.symbol_store import SymbolIndexStore
            store = SymbolIndexStore()
            stats["index"] = store.stats()
            store.close()
        except Exception:
            stats["index"] = {"error": "Not available"}

        # Try to get cache stats
        try:
            from ..cache.cache_manager import CacheManager
            cache = CacheManager()
            stats["cache"] = cache.get_stats().to_dict()
            cache.close()
        except Exception:
            stats["cache"] = {"error": "Not available"}

        if self.config.output_format == "json":
            print(json.dumps(stats, indent=2))
        else:
            self._print_header("Research Agent Statistics")
            print(f"  Version: {stats['version']}")
            print(f"  Project Root: {stats['project_root']}")
            if "index" in stats and "total" in stats["index"]:
                print(f"\n  Index:")
                print(f"    Symbols: {stats['index']['total']}")
                print(f"    Files: {stats['index']['file_count']}")
            if "cache" in stats and "hits" in stats["cache"]:
                print(f"\n  Cache:")
                print(f"    Hit Rate: {stats['cache']['hit_rate']:.1%}")

        return 0

    def _cmd_serve(self, args) -> int:
        """Handle serve command."""
        self._emit_cli_event("serve", {"host": args.host, "port": args.port})

        try:
            from ..api.research_api import ResearchAPI, APIConfig

            config = APIConfig(
                host=args.host,
                port=args.port,
                api_key=args.api_key,
            )
            api = ResearchAPI(config)

            async def run():
                await api.start()
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down...")
                await api.stop()

            asyncio.run(run())

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    def _cmd_watch(self, args) -> int:
        """Handle watch command."""
        self._emit_cli_event("watch", {"path": args.path})

        self._print_header(f"Watching: {args.path}")

        try:
            from ..feedback.incremental_updater import IncrementalUpdater, UpdaterConfig

            config = UpdaterConfig(watch_paths=[args.path])

            def on_update(event):
                print(f"  [{event.update_type.value}] {event.path}")

            updater = IncrementalUpdater(config, on_update=on_update)

            async def run():
                await updater.start()
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\nStopping...")
                await updater.stop()

            asyncio.run(run())

        except Exception as e:
            self._print_error(str(e))
            return 1

        return 0

    # ========================================================================
    # Output Helpers
    # ========================================================================

    def _print_header(self, text: str) -> None:
        """Print a header."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")

    def _print_success(self, text: str) -> None:
        """Print success message."""
        print(f"{Colors.GREEN}{text}{Colors.RESET}")

    def _print_warning(self, text: str) -> None:
        """Print warning message."""
        print(f"{Colors.YELLOW}{text}{Colors.RESET}")

    def _print_error(self, text: str) -> None:
        """Print error message."""
        print(f"{Colors.RED}Error: {text}{Colors.RESET}", file=sys.stderr)

    def _print_footer(self, text: str) -> None:
        """Print footer text."""
        print(f"\n{Colors.DIM}{text}{Colors.RESET}")

    def _print_results(self, results: List[Dict[str, Any]]) -> None:
        """Print search results."""
        if not results:
            self._print_warning("No results found")
            return

        print()
        for i, r in enumerate(results, 1):
            name = r.get("name", r.get("path", "unknown"))
            kind = r.get("kind", r.get("type", ""))
            path = r.get("path", "")
            line = r.get("line", "")
            score = r.get("score", r.get("final_score", 0))

            # Format output
            print(f"  {Colors.BOLD}{i}.{Colors.RESET} {Colors.CYAN}{name}{Colors.RESET}", end="")
            if kind:
                print(f" {Colors.DIM}({kind}){Colors.RESET}", end="")
            if score:
                print(f" {Colors.YELLOW}[{score:.2f}]{Colors.RESET}", end="")
            print()

            if path:
                loc = f"{path}"
                if line:
                    loc += f":{line}"
                print(f"     {Colors.DIM}{loc}{Colors.RESET}")

    def _print_symbol(self, symbol) -> None:
        """Print a symbol."""
        print(f"\n  {Colors.BOLD}{symbol.kind}{Colors.RESET} {Colors.CYAN}{symbol.name}{Colors.RESET}")
        print(f"    {Colors.DIM}File:{Colors.RESET} {symbol.path}:{symbol.line}")
        if symbol.signature:
            print(f"    {Colors.DIM}Signature:{Colors.RESET} {symbol.signature}")
        if symbol.docstring:
            doc = symbol.docstring.split("\n")[0][:80]
            print(f"    {Colors.DIM}Doc:{Colors.RESET} {doc}")

    def _emit_cli_event(self, command: str, data: Dict[str, Any]) -> None:
        """Emit CLI event to bus."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        import socket
        import uuid

        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": "a2a.research.cli.execute",
            "kind": "cli",
            "level": "info",
            "actor": "research-cli",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {"command": command, **data},
        }

        try:
            with open(bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass  # Don't fail on bus write errors


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> int:
    """Main entry point for Research CLI."""
    cli = ResearchCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
