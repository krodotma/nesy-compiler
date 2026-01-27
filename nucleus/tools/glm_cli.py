#!/usr/bin/env python3
"""Minimal GLM CLI using Zhipu open.bigmodel API.

Usage:
  glm_cli.py -p "prompt" [--model MODEL] [--system SYSTEM]
  glm_cli.py "prompt"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.getenv("GLM_BASE_URL") or os.getenv("MINDLIKE_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = os.getenv("GLM_MODEL")


def build_payload(prompt: str, system: str | None, model: str, max_tokens: int, temperature: float) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def fetch_models(api_key: str, base_url: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except Exception:
        return []
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return []
    models = []
    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        model_id = item.get("id")
        if model_id:
            models.append(model_id)
    return models


def choose_default_model(api_key: str, base_url: str) -> str:
    models = fetch_models(api_key, base_url)
    if not models:
        return "glm-4.6"

    def model_key(name: str) -> tuple[int, int, int]:
        # Parse glm-4.7 or glm-4.5-air -> (4,7,0)
        core = name.split("-", 1)[-1]
        core = core.split("-")[0]
        parts = core.split(".")
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (major, minor, 0)

    air_models = [m for m in models if "-air" in m]
    if air_models:
        air_models.sort(key=model_key, reverse=True)
        return air_models[0]
    models.sort(key=model_key, reverse=True)
    return models[0]


def main() -> int:
    parser = argparse.ArgumentParser(prog="glm", add_help=True)
    parser.add_argument("prompt", nargs="*", help="Prompt text")
    parser.add_argument("-p", "--prompt", dest="prompt_flag", help="Prompt text")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (default: auto)")
    parser.add_argument("--system", default=None, help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    args, _unknown = parser.parse_known_args()

    prompt = args.prompt_flag or " ".join(args.prompt).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return 2

    api_key = os.getenv("MINDLIKE_API_KEY") or os.getenv("GLM_API_KEY")
    if not api_key:
        print("MINDLIKE_API_KEY not set.", file=sys.stderr)
        return 2

    base_url = DEFAULT_BASE_URL.rstrip("/")
    model = args.model or choose_default_model(api_key, base_url)
    url = f"{base_url}/chat/completions"
    payload = build_payload(prompt, args.system, model, args.max_tokens, args.temperature)

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        print(f"GLM API error: {err.code} {err.reason}: {detail}", file=sys.stderr)
        return 1
    except Exception as err:  # noqa: BLE001
        print(f"GLM request failed: {err}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        print(data)
        return 0

    content = ""
    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content") or ""
    if content:
        print(content)
        return 0

    print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
