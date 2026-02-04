#!/usr/bin/env python3
"""Fast Codex CLI auth check (no LLM call, no subprocess)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    # Check if codex CLI exists
    codex = shutil.which("codex")
    if not codex:
        sys.stderr.write("missing codex CLI\n")
        return 2

    # Check for auth config in multiple locations
    codex_home = Path(os.environ.get("PLURIBUS_CODEX_HOME", "/pluribus/.pluribus/agent_homes/codex"))
    root_home = Path("/root")

    config_paths = [
        codex_home / ".codex" / "config.toml",
        root_home / ".codex" / "config.toml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            print(f"codex CLI: {codex}")
            print(f"config: {config_path}")
            return 0

    # Also check if codex process is running
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-f", "codex"],
            capture_output=True,
            timeout=2,
        )
        if result.returncode == 0:
            print(f"codex CLI: {codex}")
            print("auth: active session")
            return 0
    except Exception:
        pass

    sys.stderr.write("codex CLI found but no config detected\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
