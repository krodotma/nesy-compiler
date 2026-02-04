#!/usr/bin/env python3
"""Fast OpenAI/ChatGPT auth check (no LLM call).

Checks for:
1. OPENAI_API_KEY environment variable
2. Codex auth.json (contains OPENAI_API_KEY)
3. OpenAI CLI/web config
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    home = Path(os.environ.get("HOME", "/root"))

    # Check for API key in environment
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        print("openai: API key found")
        print("auth: OPENAI_API_KEY env")
        return 0

    # Check Codex auth.json (Codex uses OpenAI OAuth tokens)
    codex_auth_paths = [
        home / ".codex" / "auth.json",
        Path("/pluribus/.pluribus/agent_homes/codex/.codex/auth.json"),
    ]

    for auth_path in codex_auth_paths:
        if auth_path.exists():
            try:
                with open(auth_path) as f:
                    auth = json.load(f)
                # Check for OAuth tokens (Codex uses OAuth, not API key)
                if auth.get("tokens") and auth["tokens"].get("id_token"):
                    print("openai: via Codex OAuth")
                    print(f"auth: {auth_path}")
                    return 0
                # Also check for direct API key
                if auth.get("OPENAI_API_KEY"):
                    print("openai: via Codex API key")
                    print(f"auth: {auth_path}")
                    return 0
            except Exception:
                pass

    # Check for OpenAI CLI config
    openai_paths = [
        home / ".openai",
        home / ".config/openai",
    ]

    for config_path in openai_paths:
        if config_path.exists():
            print("openai: config found")
            print(f"auth: {config_path}")
            return 0

    # No auth found
    print("openai: no auth found")
    sys.stderr.write("No OPENAI_API_KEY or Codex auth detected\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
