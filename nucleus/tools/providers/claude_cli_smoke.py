#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="claude_cli_smoke.py", description="Claude Code CLI smoke test (web-login / setup-token).")
    p.add_argument("--prompt", required=True, help="User prompt text")
    p.add_argument("--model", default=os.environ.get("CLAUDE_CODE_MODEL") or None, help="Claude Code model alias or full name")
    p.add_argument("--json", action="store_true", help="Use --output-format json (best effort).")
    p.add_argument("--home", default=os.environ.get("PLURIBUS_CLAUDE_HOME") or None, help="Override HOME to a writable directory (stores .claude under it).")
    p.add_argument("--timeout-s", default="180", help="Hard timeout for the CLI call (default: 180).")
    args = p.parse_args(argv)

    claude = shutil.which("claude")
    if not claude:
        sys.stderr.write("missing claude CLI (install: npm i -g @anthropic-ai/claude-code)\n")
        return 2

    env = dict(os.environ)
    if args.home:
        env["HOME"] = args.home

    cmd = [claude, "--print"]
    if args.json:
        cmd += ["--output-format", "json"]
    if args.model:
        cmd += ["--model", args.model]
    cmd += [args.prompt]

    try:
        p2 = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=max(1, int(args.timeout_s)),
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"claude CLI timed out after {args.timeout_s}s\n")
        return 124
    if p2.stderr:
        sys.stderr.write(p2.stderr)
    if p2.stdout:
        sys.stdout.write(p2.stdout)
    return int(p2.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
