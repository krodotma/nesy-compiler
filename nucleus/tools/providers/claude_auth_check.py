#!/usr/bin/env python3
"""Claude CLI auth check.

This is allowed to run a *bounded* CLI invocation to avoid false positives
from stale ~/.claude folders. The goal is to prevent "always online" lies in
dashboard/TUI when Claude actually requires /login.
"""
from __future__ import annotations

import os
import shutil
import sys
import subprocess
from pathlib import Path


def main() -> int:
    # Check if claude CLI exists
    claude = shutil.which("claude")
    if not claude:
        sys.stderr.write("missing claude CLI\n")
        return 2

    # Prefer checking the dedicated Pluribus claude HOME if configured.
    default_home = "/pluribus/.pluribus/agent_homes/claude"
    home = os.environ.get("PLURIBUS_CLAUDE_HOME") or (default_home if Path(default_home).exists() else (os.environ.get("HOME") or "/root"))
    env = dict(os.environ)
    env["HOME"] = str(home)

    # Run a bounded no-op prompt; if it asks for /login, report unavailable.
    # Claude CLI startup can take 15-20s on first invocation.
    try:
        p = subprocess.run(
            [claude, "--print", "Reply with OK."],
            capture_output=True,
            text=True,
            timeout=35,
            env=env,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write("claude auth check timeout\n")
        return 1

    combined = (p.stdout or "") + "\n" + (p.stderr or "")
    if "Please run /login" in combined or "setup-token" in combined or "Invalid API key" in combined:
        sys.stderr.write("claude CLI requires /login\n")
        return 1

    if p.returncode == 0:
        print(f"claude CLI: {claude}")
        print("auth: ok")
        return 0

    sys.stderr.write((p.stderr or p.stdout or "claude auth check failed")[:200] + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
