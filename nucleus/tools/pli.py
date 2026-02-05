#!/usr/bin/env python3
"""
pli.py - Pluribus Line Interface

Unified CLI entry point for Pluribus semantic operators.
The PLI provides quick access to all PBTSO operators and verification tools.

Protocol: DKIN v29 | PAIP v15 | CITIZEN v1

Operators Available:
- verify      : PBVERIFY - System verification
- verify-tasks: PBVERIFY-TASKS - Interrupted task verification
- resume      : PBRESUME - Recovery operator
- ckin        : CKIN/DKIN - Status dashboard
- iterate     : ITERATE - Coordination kick
- swarm       : PBTSWARM - Multi-agent swarm
- loop        : PBLOOP - Ralph autonomous loop
- test        : PBTEST - Phenomenological verification
- a2a         : A2A - Agent-to-agent coordination

Usage:
    pli verify                    # Quick verification
    pli verify --full             # Full verification with tests
    pli verify-tasks              # Verify interrupted tasks
    pli verify-tasks --resolve    # Resolve stale tasks
    pli resume                    # Recovery scan
    pli resume --scope session    # Session-specific recovery
    pli ckin                      # Status dashboard
    pli swarm <manifest>          # Spawn agent swarm
    pli loop "<prompt>"           # Ralph autonomous loop
    pli a2a heartbeat             # Emit A2A heartbeat
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.dont_write_bytecode = True

# =============================================================================
# Constants
# =============================================================================

VERSION = "1.0.0"
PROTOCOL_VERSION = "DKIN v29"

TOOLS_DIR = Path(__file__).parent.resolve()
NUCLEUS_DIR = TOOLS_DIR.parent
REPO_ROOT = NUCLEUS_DIR.parent


# =============================================================================
# Helpers
# =============================================================================

def resolve_bus_dir() -> Path:
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip()
    if bus_dir:
        return Path(bus_dir).expanduser().resolve()
    root = os.environ.get("PLURIBUS_ROOT") or str(REPO_ROOT)
    return Path(root) / ".pluribus" / "bus"


def run_tool(module_path: str, args: List[str]) -> int:
    """Run a tool module with arguments."""
    full_path = TOOLS_DIR / module_path
    if not full_path.exists():
        print(f"Error: Tool not found: {full_path}", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(full_path)] + args
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


def import_and_run(module_name: str, main_func: str, args: List[str]) -> int:
    """Import a module and run its main function."""
    try:
        if str(TOOLS_DIR) not in sys.path:
            sys.path.insert(0, str(TOOLS_DIR))
        module = importlib.import_module(module_name)
        func = getattr(module, main_func)
        return func(args)
    except ImportError as e:
        print(f"Error importing {module_name}: {e}", file=sys.stderr)
        return 1
    except AttributeError:
        print(f"Error: {module_name} has no function {main_func}", file=sys.stderr)
        return 1
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0


# =============================================================================
# Operator Handlers
# =============================================================================

def cmd_verify(args: argparse.Namespace) -> int:
    """Run PBVERIFY system verification."""
    tool_args = []
    if args.full:
        tool_args.append("--full")
    if args.report:
        tool_args.append("--report")
    if args.json:
        tool_args.append("--json")
    return run_tool("pbverify_operator.py", tool_args)


def cmd_verify_tasks(args: argparse.Namespace) -> int:
    """Run PBVERIFY-TASKS for interrupted task verification."""
    tool_args = []
    if args.scope:
        tool_args.extend(["--scope", args.scope])
    if args.resolve:
        tool_args.append("--resolve-stale")
    if args.dry_run:
        tool_args.append("--dry-run")
    if args.json:
        tool_args.extend(["--format", "json"])
    tool_args.append("--emit-bus")
    return run_tool("pbverify_tasks.py", tool_args)


def cmd_resume(args: argparse.Namespace) -> int:
    """Run PBRESUME recovery operator."""
    tool_args = []
    if args.scope:
        tool_args.extend(["--scope", args.scope])
    if args.depth:
        tool_args.extend(["--depth", args.depth])
    if args.summary:
        tool_args.append("--summary")
    if args.json:
        tool_args.extend(["--format", "json"])
    if args.dry_run:
        tool_args.append("--dry-run")
    return run_tool("pbresume_operator.py", tool_args)


def cmd_ckin(args: argparse.Namespace) -> int:
    """Run CKIN status dashboard."""
    tool_args = []
    if args.agent:
        tool_args.extend(["--agent", args.agent])
    if args.json:
        tool_args.append("--json")
    if args.compact:
        tool_args.append("--compact")
    tool_args.append("--emit-bus")
    return run_tool("ckin_report.py", tool_args)


def cmd_monitor(args: argparse.Namespace) -> int:
    """Run TUI Bus Monitor."""
    tool_args = []
    if hasattr(args, 'timeout') and args.timeout:
        tool_args.extend(["--timeout", str(args.timeout)])
    return run_tool("bus_monitor_tui.py", tool_args)

def cmd_iterate(args: argparse.Namespace) -> int:
    """Run ITERATE coordination kick."""
    if args.start and args.swarm:
        # PBTSO Mode
        if str(TOOLS_DIR) not in sys.path:
            sys.path.insert(0, str(TOOLS_DIR))
        import json
        import uuid
        try:
            from agent_bus import emit_event, resolve_bus_paths
            paths = resolve_bus_paths(None)
            
            payload = {
                "swarm_id": args.swarm,
                "scope": args.scope or "default",
                "status": "STARTED"
            }
            
            emit_event(
                paths,
                topic="orchestration.iteration.start",
                kind="request",
                level="info",
                actor="pli",
                data=payload,
                trace_id=str(uuid.uuid4()),
                run_id=None,
                durable=True
            )
            print(f"PBTSO: Iteration started for {args.swarm}")
            return 0
        except ImportError as e:
            print(f"Error: agent_bus not available for PBTSO iteration: {e}", file=sys.stderr)
            return 1

    # Legacy Mode
    tool_args = []
    if args.agent:
        tool_args.extend(["--agent", args.agent])
    return run_tool("iterate_operator.py", tool_args)


def cmd_swarm(args: argparse.Namespace) -> int:
    """Run PBTSWARM multi-agent swarm."""
    tool_args = ["spawn", args.manifest] if args.manifest else ["test"]
    return run_tool("tmux_swarm_orchestrator.py", tool_args)


def cmd_loop(args: argparse.Namespace) -> int:
    """Run PBLOOP Ralph autonomous loop."""
    tool_args = ["loop", args.prompt]
    if args.max_iterations:
        tool_args.extend(["--max-iterations", str(args.max_iterations)])
    if args.promise:
        tool_args.extend(["--promise", args.promise])
    return run_tool("tmux_swarm_orchestrator.py", tool_args)


def cmd_test(args: argparse.Namespace) -> int:
    """Run PBTEST phenomenological verification."""
    tool_args = []
    if args.scope:
        tool_args.append(args.scope)
    if args.mode:
        tool_args.extend(["--mode", args.mode])
    if args.browser:
        tool_args.extend(["--browser", args.browser])
    if args.intent:
        tool_args.append(args.intent)
    return run_tool("pbtest_operator.py", tool_args)


def cmd_a2a(args: argparse.Namespace) -> int:
    """Run A2A agent-to-agent coordination."""
    # Determine the correct subcommand for tbtso_a2a.py
    if args.status:
        tool_args = ["status"]
        if args.json:
            tool_args.append("--json")
    elif args.heartbeat:
        tool_args = ["heartbeat", "default", args.agent or "pli"]
    elif args.propose:
        tool_args = ["init", "--agents", args.propose, "--scope", "handshake"]
        if args.json:
            tool_args.append("--json")
    else:
        # Default to status
        tool_args = ["status"]
        if args.json:
            tool_args.append("--json")
    return run_tool("tbtso_a2a.py", tool_args)


def cmd_affected(args: argparse.Namespace) -> int:
    """Run PBAFFECTED affected project detection."""
    tool_args = []
    if args.base:
        tool_args.extend(["--base", args.base])
    if args.head:
        tool_args.extend(["--head", args.head])
    if args.target:
        tool_args.extend(["--target", args.target])
    if args.json:
        tool_args.append("--json")
    if args.all:
        tool_args.append("--all")
    tool_args.append("--emit-bus")
    return run_tool("affected_projects.py", tool_args)


def cmd_cache(args: argparse.Namespace) -> int:
    """Run PBCACHE build cache operations."""
    subcmd = args.subcommand or "status"
    tool_args = [subcmd]
    if args.project:
        tool_args.append(args.project)
    if args.target_name:
        tool_args.append(args.target_name)
    if args.json:
        tool_args.append("--json")
    if args.force:
        tool_args.append("--force")
    # Only store and restore support --emit-bus
    if subcmd in ("store", "restore"):
        tool_args.append("--emit-bus")
    return run_tool("build_cache.py", tool_args)


def cmd_status(args: argparse.Namespace) -> int:
    """Show PLI status and available operators."""
    print(f"PLI - Pluribus Line Interface v{VERSION}")
    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Tools Dir: {TOOLS_DIR}")
    print(f"Bus Dir: {resolve_bus_dir()}")
    print()

    operators = [
        ("verify", "pbverify_operator.py", "System verification"),
        ("verify-tasks", "pbverify_tasks.py", "Interrupted task verification"),
        ("resume", "pbresume_operator.py", "Recovery operator"),
        ("ckin", "ckin_report.py", "Status dashboard"),
        ("iterate", "iterate_operator.py", "Coordination kick"),
        ("swarm", "tmux_swarm_orchestrator.py", "Multi-agent swarm"),
        ("loop", "tmux_swarm_orchestrator.py", "Ralph autonomous loop"),
        ("test", "pbtest_operator.py", "Phenomenological verification"),
        ("a2a", "tbtso_a2a.py", "Agent-to-agent coordination"),
        ("affected", "affected_projects.py", "Affected project detection"),
        ("cache", "build_cache.py", "Build artifact caching"),
    ]

    print("Operators:")
    for name, tool, desc in operators:
        path = TOOLS_DIR / tool
        status = "OK" if path.exists() else "MISSING"
        print(f"  {name:15} [{status:7}] {desc}")

    return 0


# =============================================================================
# CLI Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pli",
        description="PLI - Pluribus Line Interface: Unified CLI for Pluribus operators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pli verify                    # Quick system verification
  pli verify-tasks --resolve    # Verify and resolve stale tasks
  pli resume --summary          # Recovery scan with summary
  pli ckin                      # Status dashboard
  pli swarm manifest.json       # Spawn agent swarm
  pli loop "Fix the bug"        # Ralph autonomous loop
  pli status                    # Show PLI status
""",
    )
    parser.add_argument("--version", action="version", version=f"PLI v{VERSION}")

    subparsers = parser.add_subparsers(dest="command", help="Available operators")

    # verify
    p_verify = subparsers.add_parser("verify", help="PBVERIFY - System verification")
    p_verify.add_argument("--full", action="store_true", help="Full verification with tests")
    p_verify.add_argument("--report", action="store_true", help="Generate markdown report")
    p_verify.add_argument("--json", action="store_true", help="JSON output")
    p_verify.set_defaults(func=cmd_verify)

    # verify-tasks
    p_vtasks = subparsers.add_parser("verify-tasks", help="PBVERIFY-TASKS - Interrupted task verification")
    p_vtasks.add_argument("--scope", default="all", help="Scope filter (all, pli, a2a, rollback)")
    p_vtasks.add_argument("--resolve", action="store_true", help="Resolve stale tasks")
    p_vtasks.add_argument("--dry-run", action="store_true", help="Preview without emitting")
    p_vtasks.add_argument("--json", action="store_true", help="JSON output")
    p_vtasks.set_defaults(func=cmd_verify_tasks)

    # resume
    p_resume = subparsers.add_parser("resume", help="PBRESUME - Recovery operator")
    p_resume.add_argument("--scope", default="all", choices=["session", "lane", "all"])
    p_resume.add_argument("--depth", default="24h", help="How far back to scan")
    p_resume.add_argument("--summary", action="store_true", help="Natural language summary")
    p_resume.add_argument("--json", action="store_true", help="JSON output")
    p_resume.add_argument("--dry-run", action="store_true", help="Don't emit iteration events")
    p_resume.set_defaults(func=cmd_resume)

    # ckin
    p_ckin = subparsers.add_parser("ckin", help="CKIN/DKIN - Status dashboard")
    p_ckin.add_argument("--agent", help="Agent name for header")
    p_ckin.add_argument("--json", action="store_true", help="JSON output")
    p_ckin.add_argument("--compact", action="store_true", help="Compact output")
    p_ckin.set_defaults(func=cmd_ckin)

    # monitor
    p_monitor = subparsers.add_parser("monitor", help="PBMONITOR - TUI Bus Monitor")
    p_monitor.add_argument("--timeout", type=int, help="Timeout in seconds")
    p_monitor.set_defaults(func=cmd_monitor)

    # iterate
    p_iterate = subparsers.add_parser("iterate", help="ITERATE - Coordination kick")
    p_iterate.add_argument("--agent", help="Agent name")
    p_iterate.add_argument("--start", action="store_true", help="Start iteration")
    p_iterate.add_argument("--swarm", help="Swarm ID")
    p_iterate.add_argument("--scope", help="Iteration scope")
    p_iterate.set_defaults(func=cmd_iterate)

    # swarm
    p_swarm = subparsers.add_parser("swarm", help="PBTSWARM - Multi-agent swarm")
    p_swarm.add_argument("manifest", nargs="?", help="Manifest file for swarm")
    p_swarm.set_defaults(func=cmd_swarm)

    # loop
    p_loop = subparsers.add_parser("loop", help="PBLOOP - Ralph autonomous loop")
    p_loop.add_argument("prompt", help="Initial prompt for the loop")
    p_loop.add_argument("--max-iterations", type=int, default=50, help="Max iterations")
    p_loop.add_argument("--promise", default="DONE", help="Completion promise string")
    p_loop.set_defaults(func=cmd_loop)

    # test
    p_test = subparsers.add_parser("test", help="PBTEST - Phenomenological verification")
    p_test.add_argument("scope", nargs="?", help="File or module path")
    p_test.add_argument("--mode", choices=["unit", "live", "soak", "full"], default="live")
    p_test.add_argument("--browser", choices=["chromium", "webkit", "firefox", "laser"])
    p_test.add_argument("intent", nargs="?", help="Test intent description")
    p_test.set_defaults(func=cmd_test)

    # a2a
    p_a2a = subparsers.add_parser("a2a", help="A2A - Agent-to-agent coordination")
    p_a2a.add_argument("--agent", help="Agent identifier")
    p_a2a.add_argument("--heartbeat", action="store_true", help="Emit heartbeat")
    p_a2a.add_argument("--propose", metavar="TARGET", help="Propose handshake")
    p_a2a.add_argument("--status", action="store_true", help="Show A2A status")
    p_a2a.add_argument("--json", action="store_true", help="JSON output")
    p_a2a.set_defaults(func=cmd_a2a)

    # affected
    p_affected = subparsers.add_parser("affected", help="PBAFFECTED - Affected project detection")
    p_affected.add_argument("--base", default="origin/main", help="Base ref (default: origin/main)")
    p_affected.add_argument("--head", default="HEAD", help="Head ref (default: HEAD)")
    p_affected.add_argument("--target", help="Filter by target (build, test, lint)")
    p_affected.add_argument("--json", action="store_true", help="JSON output")
    p_affected.add_argument("--all", action="store_true", help="Show all projects")
    p_affected.set_defaults(func=cmd_affected)

    # cache
    p_cache = subparsers.add_parser("cache", help="PBCACHE - Build artifact caching")
    p_cache.add_argument("subcommand", nargs="?", choices=["hash", "check", "store", "restore", "clean", "status"],
                         help="Cache operation (hash, check, store, restore, clean, status)")
    p_cache.add_argument("project", nargs="?", help="Project ID")
    p_cache.add_argument("target_name", nargs="?", help="Target name (build, test, etc.)")
    p_cache.add_argument("--json", action="store_true", help="JSON output")
    p_cache.add_argument("--force", action="store_true", help="Force operation")
    p_cache.set_defaults(func=cmd_cache)

    # status
    p_status = subparsers.add_parser("status", help="Show PLI status and operators")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if hasattr(args, "func"):
        return args.func(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
