#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

OG_SYSTEM_PROMPT_PATH = "/pluribus/ogsystemprompt.md"


def load_system_prompt(path: str | None = None) -> str | None:
    """Load system prompt from file."""
    target = path or OG_SYSTEM_PROMPT_PATH
    try:
        with open(target, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="codex_cli_smoke.py", description="Codex CLI smoke test (non-interactive).")
    p.add_argument("--prompt", required=True, help="User prompt text")
    p.add_argument("--model", default=os.environ.get("CODEX_MODEL") or None, help="Optional Codex model name")
    p.add_argument("--cd", default=os.environ.get("PLURIBUS_ROOT") or os.getcwd(), help="Working directory for the agent session")
    p.add_argument("--system-prompt", default=None, help="Path to system prompt file")
    p.add_argument("--no-system-prompt", action="store_true", help="Disable default system prompt")
    args = p.parse_args(argv)

    # Load system prompt and prepend to user prompt for codex
    system_prompt = None
    if not args.no_system_prompt:
        system_prompt = load_system_prompt(args.system_prompt)

    effective_prompt = args.prompt
    if system_prompt:
        effective_prompt = f"[SYSTEM CONTEXT]\n{system_prompt}\n\n[USER REQUEST]\n{args.prompt}"

    codex = shutil.which("codex")
    if not codex:
        sys.stderr.write("missing codex CLI on PATH\n")
        return 2

    home = os.environ.get("PLURIBUS_CODEX_HOME") or "/pluribus/.pluribus/agent_homes/codex"
    env = dict(os.environ)
    env["HOME"] = home

    with tempfile.NamedTemporaryFile(prefix="codex_last_message_", suffix=".txt", delete=False) as tf:
        out_path = tf.name

    cmd: list[str] = [
        codex,
        "exec",
        "--skip-git-repo-check",
        "--cd",
        args.cd,
        "-s",
        "read-only",
        "--color",
        "never",
        "--output-last-message",
        out_path,
    ]
    if args.model:
        cmd += ["-m", args.model]
    cmd += [effective_prompt]

    try:
        p_run = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=120)
    except subprocess.TimeoutExpired:
        sys.stderr.write("codex CLI timed out after 120s\n")
        return 124
    if p_run.returncode != 0:
        if p_run.stderr:
            sys.stderr.write(p_run.stderr)
        else:
            sys.stderr.write(p_run.stdout)
        return int(p_run.returncode)

    try:
        text = open(out_path, "r", encoding="utf-8", errors="replace").read().strip()
    except Exception as e:
        sys.stderr.write(f"failed reading output-last-message: {e}\n")
        return 1

    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
