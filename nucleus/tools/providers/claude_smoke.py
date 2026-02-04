#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

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
    p = argparse.ArgumentParser(prog="claude_smoke.py", description="Claude smoke test via Messages API (no deps).")
    p.add_argument("--prompt", required=True, help="User prompt text")
    p.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--system-prompt", default=None, help="Path to system prompt file")
    p.add_argument("--no-system-prompt", action="store_true", help="Disable default system prompt")
    args = p.parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.stderr.write("missing ANTHROPIC_API_KEY\n")
        return 2

    # Load system prompt
    system_prompt = None
    if not args.no_system_prompt:
        system_prompt = load_system_prompt(args.system_prompt)

    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": args.prompt}],
    }
    if system_prompt:
        payload["system"] = system_prompt

    req = urllib.request.Request(
        url=url,
        method="POST",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "anthropic-version": "2023-06-01",
            "x-api-key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        sys.stderr.write(f"http error: {e.code}\n{body}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        return 1

    try:
        obj = json.loads(raw)
        chunks = obj.get("content") if isinstance(obj, dict) else None
        text = None
        if isinstance(chunks, list):
            for part in chunks:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = (part.get("text") or "").strip()
                    break
    except Exception:
        text = None

    if text:
        sys.stdout.write(text + "\n")
        return 0

    sys.stdout.write(raw + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

