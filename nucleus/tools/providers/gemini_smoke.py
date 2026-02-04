#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_loader import load_pluribus_env  # noqa: E402

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
    load_pluribus_env()
    p = argparse.ArgumentParser(prog="gemini_smoke.py", description="Gemini smoke test via REST (no deps).")
    p.add_argument("--prompt", required=True, help="User prompt text")
    p.add_argument("--model", default=os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash")
    p.add_argument("--system-prompt", default=None, help="Path to system prompt file")
    p.add_argument("--no-system-prompt", action="store_true", help="Disable default system prompt")
    args = p.parse_args(argv)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.stderr.write("missing GEMINI_API_KEY (or GOOGLE_API_KEY)\n")
        return 2

    # Load system prompt
    system_prompt = None
    if not args.no_system_prompt:
        system_prompt = load_system_prompt(args.system_prompt)

    base = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base}/models/{urllib.parse.quote(args.model)}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": args.prompt}],
            }
        ]
    }
    # Add system instruction if available
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    req = urllib.request.Request(
        url=url,
        method="POST",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-goog-api-key": api_key,
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
        text = (
            (((obj.get("candidates") or [])[0].get("content") or {}).get("parts") or [])[0].get("text")
            if isinstance(obj, dict)
            else None
        )
    except Exception:
        obj = None
        text = None

    if text:
        sys.stdout.write(text.strip() + "\n")
        return 0

    sys.stdout.write(raw + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
