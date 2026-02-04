#!/usr/bin/env python3
"""
OpenCode Wrapper: Bus-integrated wrapper for OpenCode TUI agent.
================================================================

Provides a Pluribus-native interface to OpenCode (https://github.com/sst/opencode),
emitting bus events on actions and integrating with strp_worker for delegation.

Usage:
    # Direct invocation
    python3 opencode_wrapper.py --goal "Implement feature X" --trace-id abc-123

    # As delegation target (called by strp_worker)
    python3 opencode_wrapper.py --delegate --goal "Refactor module Y"
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def emit_bus(
    bus_dir: str | None,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict,
    trace_id: str | None = None,
) -> None:
    """Emit event to the Pluribus bus."""
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    cmd = [
        sys.executable,
        str(tool),
        "--bus-dir",
        bus_dir,
        "pub",
        "--topic",
        topic,
        "--kind",
        kind,
        "--level",
        level,
        "--actor",
        actor,
        "--data",
        json.dumps(data, ensure_ascii=False),
    ]
    if trace_id:
        cmd.extend(["--trace-id", trace_id])

    subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def find_opencode() -> str | None:
    """Locate the opencode executable."""
    # Check if npx is available
    npx = shutil.which("npx")
    if npx:
        return "npx"

    # Check for global opencode binary
    opencode = shutil.which("opencode")
    if opencode:
        return opencode

    # Check common node_modules locations
    for candidate in [
        Path.home() / ".npm-global" / "bin" / "opencode",
        Path("/usr/local/bin/opencode"),
        Path("/usr/bin/opencode"),
    ]:
        if candidate.exists():
            return str(candidate)

    return None


def check_opencode_version(opencode_path: str) -> tuple[bool, str]:
    """Check if OpenCode is available and get version."""
    try:
        if opencode_path == "npx":
            cmd = ["npx", "opencode", "--version"]
        else:
            cmd = [opencode_path, "--version"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return True, version
        return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def run_opencode(
    *,
    goal: str,
    working_dir: str | None = None,
    files: list[str] | None = None,
    timeout: int = 300,
    opencode_path: str = "npx",
) -> tuple[int, str, str]:
    """
    Run OpenCode with the specified goal.

    Args:
        goal: The task/prompt for OpenCode.
        working_dir: Directory to run in.
        files: List of files to include as context.
        timeout: Max execution time in seconds.
        opencode_path: Path to opencode or 'npx'.

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    # Build command
    if opencode_path == "npx":
        cmd = ["npx", "opencode"]
    else:
        cmd = [opencode_path]

    # OpenCode uses stdin for prompts in non-interactive mode
    # We'll pass the goal via stdin
    env = os.environ.copy()
    env["OPENCODE_NON_INTERACTIVE"] = "1"

    try:
        result = subprocess.run(
            cmd,
            input=goal,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout after {timeout}s"
    except FileNotFoundError:
        return 127, "", "OpenCode not found. Install with: npm install -g opencode"
    except Exception as e:
        return 1, "", str(e)


def cmd_run(args: argparse.Namespace) -> int:
    """Execute OpenCode with bus event emission."""
    actor = args.actor or default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    trace_id = args.trace_id or os.environ.get("PLURIBUS_TRACE_ID") or str(uuid.uuid4())
    req_id = args.req_id or str(uuid.uuid4())[:8]

    # Find OpenCode
    opencode_path = find_opencode()
    if not opencode_path:
        emit_bus(
            bus_dir,
            topic="opencode.error",
            kind="response",
            level="error",
            actor=actor,
            data={
                "req_id": req_id,
                "error": "opencode_not_found",
                "message": "OpenCode not installed. Run: npm install -g opencode",
            },
            trace_id=trace_id,
        )
        print("ERROR: OpenCode not found. Install with: npm install -g opencode", file=sys.stderr)
        return 127

    # Check version
    available, version = check_opencode_version(opencode_path)
    if not available:
        emit_bus(
            bus_dir,
            topic="opencode.error",
            kind="response",
            level="error",
            actor=actor,
            data={
                "req_id": req_id,
                "error": "opencode_unavailable",
                "message": version,
            },
            trace_id=trace_id,
        )
        print(f"ERROR: OpenCode unavailable: {version}", file=sys.stderr)
        return 1

    # Emit start event
    emit_bus(
        bus_dir,
        topic="opencode.task.start",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "goal": args.goal,
            "working_dir": args.working_dir or os.getcwd(),
            "files": args.files or [],
            "version": version,
            "delegate": args.delegate,
        },
        trace_id=trace_id,
    )

    start_time = time.perf_counter()

    # Run OpenCode
    exit_code, stdout, stderr = run_opencode(
        goal=args.goal,
        working_dir=args.working_dir,
        files=args.files,
        timeout=args.timeout,
        opencode_path=opencode_path,
    )

    duration = time.perf_counter() - start_time

    # Parse output for artifacts/changes
    artifacts = []
    files_modified = []

    # Try to detect file modifications from output
    for line in stdout.split("\n"):
        line_lower = line.lower()
        if "wrote" in line_lower or "created" in line_lower or "modified" in line_lower:
            # Heuristic: extract file paths
            parts = line.split()
            for part in parts:
                if "/" in part or part.endswith(".py") or part.endswith(".ts") or part.endswith(".js"):
                    files_modified.append(part.strip("'\""))

    # Emit completion event
    status = "success" if exit_code == 0 else "error"
    emit_bus(
        bus_dir,
        topic="opencode.task.complete",
        kind="response",
        level="info" if exit_code == 0 else "error",
        actor=actor,
        data={
            "req_id": req_id,
            "status": status,
            "exit_code": exit_code,
            "duration_s": round(duration, 2),
            "files_modified": files_modified,
            "artifacts": artifacts,
            "output_lines": len(stdout.split("\n")) if stdout else 0,
            "error": stderr.strip() if stderr else None,
        },
        trace_id=trace_id,
    )

    # Output result
    if args.json_output:
        result = {
            "req_id": req_id,
            "trace_id": trace_id,
            "status": status,
            "exit_code": exit_code,
            "duration_s": round(duration, 2),
            "stdout": stdout,
            "stderr": stderr,
            "files_modified": files_modified,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if stdout:
            print(stdout)
        if stderr and exit_code != 0:
            print(stderr, file=sys.stderr)

    return exit_code


def cmd_check(args: argparse.Namespace) -> int:
    """Check if OpenCode is available."""
    opencode_path = find_opencode()
    if not opencode_path:
        print("OpenCode: NOT FOUND")
        print("Install with: npm install -g opencode")
        return 1

    available, version = check_opencode_version(opencode_path)
    if available:
        print(f"OpenCode: OK (version: {version})")
        print(f"Path: {opencode_path}")
        return 0
    else:
        print(f"OpenCode: ERROR ({version})")
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="opencode_wrapper.py",
        description="Bus-integrated wrapper for OpenCode TUI agent",
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    p.add_argument("--actor", default=None, help="Actor name for bus events")

    sub = p.add_subparsers(dest="cmd")

    # Run command
    run_p = sub.add_parser("run", help="Run OpenCode with a goal")
    run_p.add_argument("--goal", required=True, help="Task description for OpenCode")
    run_p.add_argument("--working-dir", default=None, help="Working directory")
    run_p.add_argument("--files", nargs="*", default=None, help="Files to include as context")
    run_p.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    run_p.add_argument("--trace-id", default=None, help="Trace ID for correlation")
    run_p.add_argument("--req-id", default=None, help="Request ID")
    run_p.add_argument("--delegate", action="store_true", help="Mark as delegated task")
    run_p.add_argument("--json-output", action="store_true", help="Output JSON result")
    run_p.set_defaults(func=cmd_run)

    # Check command
    check_p = sub.add_parser("check", help="Check if OpenCode is available")
    check_p.set_defaults(func=cmd_check)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return 0

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
