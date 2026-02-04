#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
import subprocess

from env_loader import load_pluribus_env


def _present(value: str | None) -> bool:
    return bool(value and value.strip())

def _node_major(cmd: str) -> int | None:
    try:
        p = subprocess.run([cmd, "-v"], check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2)
        if p.returncode != 0:
            return None
        s = (p.stdout or "").strip()
        if s.startswith("v"):
            s = s[1:]
        return int(s.split(".", 1)[0])
    except Exception:
        return None

def main() -> int:
    load_pluribus_env()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    sys.stdout.write("nucleus llm doctor\n\n")

    sys.stdout.write("Gemini:\n")
    sys.stdout.write(f"- key: {'set' if _present(gemini_key) else 'missing'} (GEMINI_API_KEY or GOOGLE_API_KEY)\n")
    sys.stdout.write(f"- model: {os.environ.get('GEMINI_MODEL') or 'gemini-2.0-flash'} (GEMINI_MODEL)\n\n")

    sys.stdout.write("Claude:\n")
    sys.stdout.write(f"- key: {'set' if _present(anthropic_key) else 'missing'} (ANTHROPIC_API_KEY)\n")
    sys.stdout.write(f"- model: {os.environ.get('ANTHROPIC_MODEL') or 'claude-3-5-sonnet-20241022'} (ANTHROPIC_MODEL)\n")
    claude_cli = shutil.which("claude")
    sys.stdout.write(f"- claude CLI: {'found at ' + claude_cli if claude_cli else 'not found'}\n")
    if claude_cli:
        sys.stdout.write("- claude web-login: run `claude setup-token`\n")
    sys.stdout.write("\n")

    sys.stdout.write("Codex:\n")
    sys.stdout.write(f"- key: {'set' if _present(openai_key) else 'missing'} (OPENAI_API_KEY)\n\n")

    sys.stdout.write("CLI tooling:\n")
    node = shutil.which("node")
    node_major = _node_major(node) if node else None
    sys.stdout.write(f"- node: {node_major if node_major is not None else 'missing/unknown'}\n")
    gemini_cli = shutil.which("gemini")
    sys.stdout.write(f"- gemini CLI: {'found at ' + gemini_cli if gemini_cli else 'not found'}\n")
    if gemini_cli and (node_major or 0) < 20:
        node20 = Path(os.environ.get('HOME') or '~').expanduser() / ".local" / "node20" / "bin" / "node"
        if node20.exists():
            sys.stdout.write("- gemini CLI note: Node < 20 on PATH; prefix PATH with ~/.local/node20/bin to run gemini\n")
        else:
            sys.stdout.write("- gemini CLI note: requires Node >= 20\n")
    sys.stdout.write("\n")

    sys.stdout.write("Docs:\n")
    sys.stdout.write("- docs/providers/README.md\n")
    sys.stdout.write("- docs/providers/gemini.md\n")
    sys.stdout.write("- docs/providers/claude.md\n")
    sys.stdout.write("- tools/mesh_status.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
