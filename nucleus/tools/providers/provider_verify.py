#!/usr/bin/env python3
"""
Provider verification with random prompt/response.
Returns model info + actual response for verification.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Random verification prompts - short, unique responses expected
VERIFY_PROMPTS = [
    "Reply with exactly 3 random words.",
    "Say a number between 1 and 100.",
    "Name any color and stop.",
    "Complete: The quick brown ___",
    "What is 7 times 8?",
    "Say hello in any language.",
    "Name one planet.",
    "What day comes after Monday?",
    "Say any fruit name.",
    "Count to 3.",
]


def verify_claude(timeout: int = 30) -> dict:
    """Verify Claude CLI with prompt/response."""
    claude = shutil.which("claude")
    if not claude:
        return {"available": False, "error": "missing claude CLI"}

    prompt = random.choice(VERIFY_PROMPTS)

    try:
        result = subprocess.run(
            [claude, "--print", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            response = result.stdout.strip()[:200]  # Limit response length
            return {
                "available": True,
                "model": "claude-code",
                "prompt": prompt,
                "response": response,
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        else:
            return {
                "available": False,
                "error": result.stderr[:100] or "unknown error",
            }
    except subprocess.TimeoutExpired:
        return {"available": False, "error": f"timeout ({timeout}s)"}
    except Exception as e:
        return {"available": False, "error": str(e)[:100]}


def verify_codex(timeout: int = 60) -> dict:
    """Verify Codex CLI with prompt/response."""
    codex = shutil.which("codex")
    if not codex:
        return {"available": False, "error": "missing codex CLI"}

    prompt = random.choice(VERIFY_PROMPTS)
    codex_home = os.environ.get("PLURIBUS_CODEX_HOME", "/pluribus/.pluribus/agent_homes/codex")
    env = dict(os.environ)
    env["HOME"] = codex_home

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(prefix="codex_verify_", suffix=".txt", delete=False) as tf:
            out_path = tf.name

        result = subprocess.run(
            [
                codex, "exec",
                "--skip-git-repo-check",
                "--cd", "/pluribus",
                "-s", "read-only",
                "--color", "never",
                "--output-last-message", out_path,
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        if result.returncode == 0:
            try:
                response = open(out_path, "r").read().strip()[:200]
            except Exception:
                response = result.stdout.strip()[:200]
            return {
                "available": True,
                "model": "codex-cli (gpt-5.2)",
                "prompt": prompt,
                "response": response,
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        else:
            return {
                "available": False,
                "error": (result.stderr or result.stdout)[:100],
            }
    except subprocess.TimeoutExpired:
        return {"available": False, "error": f"timeout ({timeout}s)"}
    except Exception as e:
        return {"available": False, "error": str(e)[:100]}


def verify_gemini(timeout: int = 30) -> dict:
    """Verify Gemini CLI with prompt/response."""
    gemini = shutil.which("gemini")
    if not gemini:
        return {"available": False, "error": "missing gemini CLI"}

    prompt = random.choice(VERIFY_PROMPTS)

    try:
        result = subprocess.run(
            [gemini, "--output-format", "text", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            response = result.stdout.strip()[:200]
            return {
                "available": True,
                "model": "gemini-cli",
                "prompt": prompt,
                "response": response,
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        else:
            error = result.stderr[:100]
            # Check for quota error
            if "429" in error or "quota" in error.lower():
                return {"available": False, "error": "quota exceeded (429)"}
            return {"available": False, "error": error or "unknown error"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": f"timeout ({timeout}s)"}
    except Exception as e:
        return {"available": False, "error": str(e)[:100]}


def verify_mock(timeout: int = 1) -> dict:
    """Verify mock provider (always works)."""
    prompt = random.choice(VERIFY_PROMPTS)
    return {
        "available": True,
        "model": "mock-v1",
        "prompt": prompt,
        "response": f"[MOCK] Received: {prompt[:30]}...",
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify provider with prompt/response")
    parser.add_argument("provider", choices=["claude", "codex", "gemini", "mock", "all"])
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    verifiers = {
        "claude": verify_claude,
        "codex": verify_codex,
        "gemini": verify_gemini,
        "mock": verify_mock,
    }

    if args.provider == "all":
        results = {}
        for name, fn in verifiers.items():
            results[name] = fn(args.timeout)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for name, result in results.items():
                status = "✓" if result["available"] else "✗"
                print(f"{status} {name}: {result.get('model', result.get('error', 'unknown'))}")
                if result.get("response"):
                    print(f"   prompt: {result['prompt']}")
                    print(f"   response: {result['response'][:80]}...")
    else:
        fn = verifiers[args.provider]
        result = fn(args.timeout)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            status = "✓" if result["available"] else "✗"
            print(f"{status} {args.provider}: {result.get('model', result.get('error', 'unknown'))}")
            if result.get("response"):
                print(f"   prompt: {result['prompt']}")
                print(f"   response: {result['response']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
