#!/usr/bin/env python3
"""
ollama_smoke.py - Ollama local inference provider for Pluribus

Connects to a local Ollama server for inference.
Expects Ollama to be running on localhost:11434.

Usage:
    python ollama_smoke.py --prompt "Your prompt here"
    python ollama_smoke.py --prompt "Your prompt" --model "llama3.2"
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.request
import urllib.error

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OG_SYSTEM_PROMPT_PATH = "/pluribus/ogsystemprompt.md"


def load_system_prompt(path: str | None = None) -> str | None:
    """Load system prompt from file."""
    target = path or OG_SYSTEM_PROMPT_PATH
    try:
        with open(target, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_available_models() -> list[str]:
    """Get list of available models from Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            models = data.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
    except Exception:
        return []


def infer(prompt: str, model: str | None = None, max_tokens: int = 2048, temperature: float = 0.7, system_prompt: str | None = None) -> str:
    """Run inference on Ollama server."""
    if not check_ollama_available():
        raise RuntimeError("Ollama server not available at " + OLLAMA_BASE_URL)

    # Determine model
    if not model:
        available = get_available_models()
        if available:
            # Prefer common models
            for preferred in ["llama3.2", "llama3.1", "qwen2.5-coder", "mistral"]:
                for m in available:
                    if preferred in m.lower():
                        model = m
                        break
                if model:
                    break
            if not model:
                model = available[0]
        else:
            model = OLLAMA_DEFAULT_MODEL

    # Build messages with optional system prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Ollama uses /api/chat for chat completions
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            message = result.get("message", {})
            return message.get("content", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"Ollama error {e.code}: {body}")


def main():
    parser = argparse.ArgumentParser(description="Ollama local inference smoke test")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if not specified)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system-prompt", default=None, help="Path to system prompt file (default: /pluribus/ogsystemprompt.md)")
    parser.add_argument("--no-system-prompt", action="store_true", help="Disable default system prompt")
    args = parser.parse_args()

    # Load system prompt
    system_prompt = None
    if not args.no_system_prompt:
        system_prompt = load_system_prompt(args.system_prompt)

    try:
        response = infer(
            args.prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=system_prompt,
        )
        print(response)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
