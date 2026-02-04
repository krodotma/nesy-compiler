#!/usr/bin/env python3
"""
PR-Agent Wrapper: Bus-integrated wrapper for PR-Agent code review.
==================================================================

Provides a Pluribus-native interface to PR-Agent (https://github.com/Codium-ai/pr-agent),
emitting bus events on reviews and integrating with git hooks for automatic PR analysis.

Features:
- /review: Full PR review
- /improve: Suggest improvements
- /describe: Generate PR description
- /test: Suggest test cases

Usage:
    # Review a PR
    python3 pr_agent_wrapper.py review --pr-url https://github.com/org/repo/pull/123

    # Run from git hook
    python3 pr_agent_wrapper.py hook --event pull_request.opened

    # Check installation
    python3 pr_agent_wrapper.py check
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
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


def find_pr_agent() -> tuple[str | None, str]:
    """
    Locate the pr-agent executable or module.

    Returns:
        Tuple of (path_or_method, install_type)
        install_type: 'cli' | 'module' | 'docker' | None
    """
    # Check for CLI binary
    pr_agent_cli = shutil.which("pr-agent")
    if pr_agent_cli:
        return pr_agent_cli, "cli"

    # Check for Python module
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import pr_agent; print(pr_agent.__file__)"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip(), "module"
    except Exception:
        pass

    # Check for Docker
    docker = shutil.which("docker")
    if docker:
        result = subprocess.run(
            ["docker", "images", "-q", "codiumai/pr-agent"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "docker", "docker"

    return None, "none"


def check_pr_agent_version(path: str, install_type: str) -> tuple[bool, str]:
    """Check if PR-Agent is available and get version."""
    try:
        if install_type == "cli":
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        elif install_type == "module":
            result = subprocess.run(
                [sys.executable, "-m", "pr_agent", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        elif install_type == "docker":
            result = subprocess.run(
                ["docker", "run", "--rm", "codiumai/pr-agent", "--version"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        else:
            return False, "unknown install type"

        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip() or "unknown"
            return True, version
        # PR-Agent might not have --version, try --help
        return True, "installed (version unknown)"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def parse_pr_url(url: str) -> dict | None:
    """Parse GitHub PR URL into components."""
    # https://github.com/owner/repo/pull/123
    pattern = r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.match(pattern, url)
    if match:
        return {
            "owner": match.group(1),
            "repo": match.group(2),
            "pr_number": int(match.group(3)),
            "provider": "github",
        }

    # GitLab: https://gitlab.com/owner/repo/-/merge_requests/123
    pattern_gl = r"https?://gitlab\.com/([^/]+)/([^/]+)/-/merge_requests/(\d+)"
    match_gl = re.match(pattern_gl, url)
    if match_gl:
        return {
            "owner": match_gl.group(1),
            "repo": match_gl.group(2),
            "pr_number": int(match_gl.group(3)),
            "provider": "gitlab",
        }

    return None


def run_pr_agent_command(
    *,
    command: str,
    pr_url: str,
    install_type: str,
    path: str,
    extra_args: list[str] | None = None,
    timeout: int = 300,
) -> tuple[int, str, str]:
    """
    Run a PR-Agent command.

    Args:
        command: review, improve, describe, test
        pr_url: Full PR URL
        install_type: cli, module, docker
        path: Path to executable/module
        extra_args: Additional arguments
        timeout: Max execution time

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    extra = extra_args or []

    # Build command based on install type
    if install_type == "cli":
        cmd = [path, f"--{command}", pr_url] + extra
    elif install_type == "module":
        cmd = [sys.executable, "-m", "pr_agent", f"--{command}", pr_url] + extra
    elif install_type == "docker":
        # Docker requires API keys to be passed
        env_args = []
        for key in ["OPENAI_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN", "ANTHROPIC_API_KEY"]:
            if os.environ.get(key):
                env_args.extend(["-e", f"{key}={os.environ[key]}"])

        cmd = ["docker", "run", "--rm"] + env_args + ["codiumai/pr-agent", f"--{command}", pr_url] + extra
    else:
        return 1, "", "Invalid install type"

    env = os.environ.copy()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout after {timeout}s"
    except FileNotFoundError:
        return 127, "", "PR-Agent not found"
    except Exception as e:
        return 1, "", str(e)


def cmd_review(args: argparse.Namespace) -> int:
    """Run PR review."""
    return _run_pr_command(args, "review")


def cmd_improve(args: argparse.Namespace) -> int:
    """Run PR improvement suggestions."""
    return _run_pr_command(args, "improve")


def cmd_describe(args: argparse.Namespace) -> int:
    """Generate PR description."""
    return _run_pr_command(args, "describe")


def cmd_test(args: argparse.Namespace) -> int:
    """Suggest test cases."""
    return _run_pr_command(args, "test")


def _run_pr_command(args: argparse.Namespace, command: str) -> int:
    """Generic PR command runner with bus integration."""
    actor = args.actor or default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    trace_id = args.trace_id or os.environ.get("PLURIBUS_TRACE_ID") or str(uuid.uuid4())
    req_id = args.req_id or str(uuid.uuid4())[:8]

    # Parse PR URL
    pr_info = parse_pr_url(args.pr_url)
    if not pr_info:
        emit_bus(
            bus_dir,
            topic="pr_agent.error",
            kind="response",
            level="error",
            actor=actor,
            data={
                "req_id": req_id,
                "error": "invalid_pr_url",
                "pr_url": args.pr_url,
            },
            trace_id=trace_id,
        )
        print(f"ERROR: Invalid PR URL: {args.pr_url}", file=sys.stderr)
        return 1

    # Find PR-Agent
    path, install_type = find_pr_agent()
    if not path:
        emit_bus(
            bus_dir,
            topic="pr_agent.error",
            kind="response",
            level="error",
            actor=actor,
            data={
                "req_id": req_id,
                "error": "pr_agent_not_found",
                "message": "PR-Agent not installed. Run: pip install pr-agent",
            },
            trace_id=trace_id,
        )
        print("ERROR: PR-Agent not found. Install with: pip install pr-agent", file=sys.stderr)
        return 127

    # Emit start event
    emit_bus(
        bus_dir,
        topic=f"pr_agent.{command}.start",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "command": command,
            "pr_url": args.pr_url,
            **pr_info,
            "install_type": install_type,
        },
        trace_id=trace_id,
    )

    start_time = time.perf_counter()

    # Run PR-Agent
    exit_code, stdout, stderr = run_pr_agent_command(
        command=command,
        pr_url=args.pr_url,
        install_type=install_type,
        path=path,
        timeout=args.timeout,
    )

    duration = time.perf_counter() - start_time

    # Parse output for structured data
    review_data = {
        "summary": None,
        "suggestions": [],
        "issues_found": 0,
        "score": None,
    }

    # Try to extract structured info from output
    if stdout:
        lines = stdout.split("\n")
        for line in lines:
            if "score" in line.lower():
                # Try to extract numeric score
                numbers = re.findall(r"\d+(?:\.\d+)?", line)
                if numbers:
                    review_data["score"] = float(numbers[0])
            if "issue" in line.lower() or "bug" in line.lower() or "error" in line.lower():
                review_data["issues_found"] += 1

    # Emit completion event
    status = "success" if exit_code == 0 else "error"
    emit_bus(
        bus_dir,
        topic=f"pr_agent.{command}.complete",
        kind="response",
        level="info" if exit_code == 0 else "error",
        actor=actor,
        data={
            "req_id": req_id,
            "status": status,
            "command": command,
            "exit_code": exit_code,
            "duration_s": round(duration, 2),
            **pr_info,
            **review_data,
            "error": stderr.strip() if stderr else None,
        },
        trace_id=trace_id,
    )

    # Output result
    if args.json_output:
        result = {
            "req_id": req_id,
            "trace_id": trace_id,
            "command": command,
            "status": status,
            "exit_code": exit_code,
            "duration_s": round(duration, 2),
            "pr_info": pr_info,
            "stdout": stdout,
            "stderr": stderr,
            **review_data,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if stdout:
            print(stdout)
        if stderr and exit_code != 0:
            print(stderr, file=sys.stderr)

    return exit_code


def cmd_hook(args: argparse.Namespace) -> int:
    """Handle git hook event (for automatic PR analysis)."""
    actor = args.actor or default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    trace_id = args.trace_id or os.environ.get("PLURIBUS_TRACE_ID") or str(uuid.uuid4())
    req_id = str(uuid.uuid4())[:8]

    event = args.event
    pr_url = args.pr_url

    # Map events to commands
    event_commands = {
        "pull_request.opened": "review",
        "pull_request.synchronize": "review",
        "issue_comment.created": None,  # Parse comment for /command
    }

    command = event_commands.get(event)

    if event == "issue_comment.created":
        # Check comment text for slash commands
        comment = args.comment or ""
        if "/review" in comment:
            command = "review"
        elif "/improve" in comment:
            command = "improve"
        elif "/describe" in comment:
            command = "describe"
        elif "/test" in comment:
            command = "test"

    if not command:
        emit_bus(
            bus_dir,
            topic="pr_agent.hook.skip",
            kind="log",
            level="debug",
            actor=actor,
            data={
                "req_id": req_id,
                "event": event,
                "reason": "no_matching_command",
            },
            trace_id=trace_id,
        )
        return 0

    if not pr_url:
        emit_bus(
            bus_dir,
            topic="pr_agent.hook.error",
            kind="response",
            level="error",
            actor=actor,
            data={
                "req_id": req_id,
                "event": event,
                "error": "missing_pr_url",
            },
            trace_id=trace_id,
        )
        print("ERROR: --pr-url required for hook", file=sys.stderr)
        return 1

    # Emit hook trigger event
    emit_bus(
        bus_dir,
        topic="pr_agent.hook.trigger",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "event": event,
            "command": command,
            "pr_url": pr_url,
        },
        trace_id=trace_id,
    )

    # Create a fake args object for the command runner
    class HookArgs:
        pass

    hook_args = HookArgs()
    hook_args.pr_url = pr_url
    hook_args.actor = actor
    hook_args.bus_dir = bus_dir
    hook_args.trace_id = trace_id
    hook_args.req_id = req_id
    hook_args.timeout = args.timeout
    hook_args.json_output = args.json_output

    return _run_pr_command(hook_args, command)


def cmd_check(args: argparse.Namespace) -> int:
    """Check if PR-Agent is available."""
    path, install_type = find_pr_agent()

    if not path:
        print("PR-Agent: NOT FOUND")
        print()
        print("Install options:")
        print("  pip install pr-agent")
        print("  docker pull codiumai/pr-agent")
        return 1

    available, version = check_pr_agent_version(path, install_type)

    if available:
        print(f"PR-Agent: OK")
        print(f"Install type: {install_type}")
        print(f"Version: {version}")
        if install_type != "docker":
            print(f"Path: {path}")
        return 0
    else:
        print(f"PR-Agent: ERROR ({version})")
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pr_agent_wrapper.py",
        description="Bus-integrated wrapper for PR-Agent code review",
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    p.add_argument("--actor", default=None, help="Actor name for bus events")

    sub = p.add_subparsers(dest="cmd")

    # Review command
    review_p = sub.add_parser("review", help="Run full PR review")
    review_p.add_argument("--pr-url", required=True, help="PR URL")
    review_p.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    review_p.add_argument("--trace-id", default=None, help="Trace ID")
    review_p.add_argument("--req-id", default=None, help="Request ID")
    review_p.add_argument("--json-output", action="store_true", help="Output JSON")
    review_p.set_defaults(func=cmd_review)

    # Improve command
    improve_p = sub.add_parser("improve", help="Suggest improvements")
    improve_p.add_argument("--pr-url", required=True, help="PR URL")
    improve_p.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    improve_p.add_argument("--trace-id", default=None, help="Trace ID")
    improve_p.add_argument("--req-id", default=None, help="Request ID")
    improve_p.add_argument("--json-output", action="store_true", help="Output JSON")
    improve_p.set_defaults(func=cmd_improve)

    # Describe command
    describe_p = sub.add_parser("describe", help="Generate PR description")
    describe_p.add_argument("--pr-url", required=True, help="PR URL")
    describe_p.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    describe_p.add_argument("--trace-id", default=None, help="Trace ID")
    describe_p.add_argument("--req-id", default=None, help="Request ID")
    describe_p.add_argument("--json-output", action="store_true", help="Output JSON")
    describe_p.set_defaults(func=cmd_describe)

    # Test command
    test_p = sub.add_parser("test", help="Suggest test cases")
    test_p.add_argument("--pr-url", required=True, help="PR URL")
    test_p.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    test_p.add_argument("--trace-id", default=None, help="Trace ID")
    test_p.add_argument("--req-id", default=None, help="Request ID")
    test_p.add_argument("--json-output", action="store_true", help="Output JSON")
    test_p.set_defaults(func=cmd_test)

    # Hook command (for git hooks)
    hook_p = sub.add_parser("hook", help="Handle git hook event")
    hook_p.add_argument("--event", required=True, help="Git event type (e.g., pull_request.opened)")
    hook_p.add_argument("--pr-url", default=None, help="PR URL")
    hook_p.add_argument("--comment", default=None, help="Comment text (for issue_comment events)")
    hook_p.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    hook_p.add_argument("--trace-id", default=None, help="Trace ID")
    hook_p.add_argument("--json-output", action="store_true", help="Output JSON")
    hook_p.set_defaults(func=cmd_hook)

    # Check command
    check_p = sub.add_parser("check", help="Check if PR-Agent is available")
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
