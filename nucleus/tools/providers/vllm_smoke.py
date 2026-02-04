#!/usr/bin/env python3
"""
vllm_smoke.py - vLLM local inference provider for Pluribus

Connects to a local vLLM server (OpenAI-compatible API) for inference.
Expects vLLM to be running on localhost:8000.

Usage:
    python vllm_smoke.py --prompt "Your prompt here"
    python vllm_smoke.py --prompt "Your prompt" --model "meta-llama/Llama-3.1-8B-Instruct"
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.request
import urllib.error

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
OG_SYSTEM_PROMPT_PATH = "/pluribus/ogsystemprompt.md"


def load_system_prompt(path: str | None = None) -> str | None:
    """Load system prompt from file."""
    target = path or OG_SYSTEM_PROMPT_PATH
    try:
        with open(target, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def check_vllm_available() -> bool:
    """Check if vLLM server is running."""
    try:
        req = urllib.request.Request(
            f"{VLLM_BASE_URL.rstrip('/v1')}/health",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_available_model() -> str | None:
    """Get first available model from vLLM."""
    try:
        req = urllib.request.Request(f"{VLLM_BASE_URL}/models", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            if data.get("data"):
                return data["data"][0].get("id")
    except Exception:
        pass
    return None


def infer(prompt: str, model: str | None = None, max_tokens: int = 2048, temperature: float = 0.7, system_prompt: str | None = None) -> str:
    """Run inference on vLLM server."""
    if not check_vllm_available():
        raise RuntimeError("vLLM server not available at " + VLLM_BASE_URL)

    # Determine model
    if not model:
        model = get_available_model() or VLLM_DEFAULT_MODEL

    # Build messages with optional system prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{VLLM_BASE_URL}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"vLLM error {e.code}: {body}")


def main():
    parser = argparse.ArgumentParser(description="vLLM local inference smoke test")
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
