#!/usr/bin/env python3
"""
local_router.py - Unified Local LLM Router for Pluribus

Provides LiteLLM-based routing to local inference backends:
- vLLM (high-throughput quantized inference)
- Ollama (easy local models)
- WebLLM (browser-based, signaled via bus)

All cloud providers (GPT, Claude, Gemini) remain headless browser-based.
This router handles LOCAL inference only.

Usage:
    python -m nucleus.tools.local_router start
    python -m nucleus.tools.local_router status
    python -m nucleus.tools.local_router infer "prompt"
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

# Bus integration
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
EVENTS_PATH = BUS_DIR / "events.ndjson"
SESSION_PATH = Path("/pluribus/.pluribus/vps_session.json")
LOCAL_ROUTER_STATE = Path("/pluribus/.pluribus/local_router_state.json")

# Default ports
LITELLM_PORT = 4000
VLLM_PORT = 8000
OLLAMA_PORT = 11434

# Default models
DEFAULT_VLLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OLLAMA_MODEL = "llama3.2"


@dataclass
class LocalProvider:
    """Status of a local inference provider."""
    name: str
    available: bool = False
    base_url: str = ""
    model: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None
    checked_at: str = ""


@dataclass
class RouterState:
    """State of the local router."""
    litellm_running: bool = False
    litellm_pid: Optional[int] = None
    providers: dict = field(default_factory=dict)
    fallback_order: list = field(default_factory=list)
    privacy_mode: bool = False
    started_at: str = ""
    last_check: str = ""


def emit_bus_event(topic: str, kind: str, data: dict, level: str = "info"):
    """Emit event to Pluribus bus."""
    if not EVENTS_PATH.parent.exists():
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "id": f"{int(time.time() * 1000)}-{os.urandom(4).hex()}",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": "local-router",
        "ts": time.time(),
        "iso": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }

    with open(EVENTS_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def check_http_health(base_url: str, endpoint: str = "/health", timeout: float = 2.0) -> tuple[bool, float, Optional[str]]:
    """Check if HTTP endpoint is healthy. Returns (available, latency_ms, error)."""
    import urllib.request
    import urllib.error

    # Ensure we use IP to avoid resolution issues
    url = f"{base_url.rstrip('/')}{endpoint}".replace("localhost", "127.0.0.1")
    start = time.time()

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency = (time.time() - start) * 1000
            return resp.status == 200, latency, None
    except Exception as e:
        return False, 0.0, str(e)


def check_vllm() -> LocalProvider:
    """Check vLLM server availability."""
    base_url = f"http://127.0.0.1:{VLLM_PORT}"
    available, latency, error = check_http_health(base_url, "/health")

    model = ""
    if available:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as resp:
                data = json.loads(resp.read())
                if data.get("data"):
                    model = data["data"][0].get("id", DEFAULT_VLLM_MODEL)
        except Exception:
            model = DEFAULT_VLLM_MODEL

    return LocalProvider(
        name="vllm-local",
        available=available,
        base_url=f"{base_url}/v1" if available else "",
        model=model,
        latency_ms=latency,
        error=error,
        checked_at=datetime.utcnow().isoformat() + "Z",
    )


def check_ollama() -> LocalProvider:
    """Check Ollama server availability."""
    base_url = f"http://127.0.0.1:{OLLAMA_PORT}"
    available, latency, error = check_http_health(base_url, "/api/tags")

    model = ""
    if available:
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
                data = json.loads(resp.read())
                models = data.get("models", [])
                if models:
                    model = models[0].get("name", DEFAULT_OLLAMA_MODEL)
        except Exception:
            model = DEFAULT_OLLAMA_MODEL

    return LocalProvider(
        name="ollama-local",
        available=available,
        base_url=base_url if available else "",
        model=model,
        latency_ms=latency,
        error=error,
        checked_at=datetime.utcnow().isoformat() + "Z",
    )


def check_litellm_proxy() -> LocalProvider:
    """Check LiteLLM proxy availability."""
    base_url = f"http://127.0.0.1:{LITELLM_PORT}"
    available, latency, error = check_http_health(base_url, "/health")

    return LocalProvider(
        name="litellm-proxy",
        available=available,
        base_url=f"{base_url}/v1" if available else "",
        model="router",
        latency_ms=latency,
        error=error,
        checked_at=datetime.utcnow().isoformat() + "Z",
    )


def get_router_state() -> RouterState:
    """Get current router state, checking all providers."""
    state = RouterState()

    # Check providers
    vllm = check_vllm()
    ollama = check_ollama()
    litellm = check_litellm_proxy()

    state.providers = {
        "vllm-local": asdict(vllm),
        "ollama-local": asdict(ollama),
        "litellm-proxy": asdict(litellm),
    }

    state.litellm_running = litellm.available

    # Build fallback order based on availability
    state.fallback_order = []
    if vllm.available:
        state.fallback_order.append("vllm-local")
    if ollama.available:
        state.fallback_order.append("ollama-local")
    # WebLLM is browser-side, always "available" as fallback
    state.fallback_order.append("webllm-browser")
    state.fallback_order.append("mock")

    # Check privacy mode from session
    try:
        if SESSION_PATH.exists():
            with open(SESSION_PATH) as f:
                session = json.load(f)
                state.privacy_mode = session.get("privacy_mode", False)
    except Exception:
        pass

    state.last_check = datetime.utcnow().isoformat() + "Z"

    return state


def save_state(state: RouterState):
    """Save router state to file."""
    LOCAL_ROUTER_STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_ROUTER_STATE, "w") as f:
        json.dump(asdict(state), f, indent=2)


def load_state() -> Optional[RouterState]:
    """Load saved router state."""
    if not LOCAL_ROUTER_STATE.exists():
        return None
    try:
        with open(LOCAL_ROUTER_STATE) as f:
            data = json.load(f)
            return RouterState(**data)
    except Exception:
        return None


def generate_litellm_config() -> dict:
    """Generate LiteLLM proxy configuration."""
    state = get_router_state()

    model_list = []

    # Add vLLM if available
    if state.providers.get("vllm-local", {}).get("available"):
        vllm = state.providers["vllm-local"]
        model_list.append({
            "model_name": "vllm-local",
            "litellm_params": {
                "model": f"openai/{vllm.get('model', DEFAULT_VLLM_MODEL)}",
                "api_base": vllm.get("base_url", f"http://localhost:{VLLM_PORT}/v1"),
                "api_key": "not-needed",
            }
        })

    # Add Ollama if available
    if state.providers.get("ollama-local", {}).get("available"):
        ollama = state.providers["ollama-local"]
        model_list.append({
            "model_name": "ollama-local",
            "litellm_params": {
                "model": f"ollama/{ollama.get('model', DEFAULT_OLLAMA_MODEL)}",
                "api_base": ollama.get("base_url", f"http://localhost:{OLLAMA_PORT}"),
            }
        })

    # Fallback settings
    router_settings = {
        "routing_strategy": "simple-shuffle",
        "num_retries": 2,
        "timeout": 120,
        "fallbacks": [],
    }

    if len(model_list) >= 2:
        router_settings["fallbacks"] = [
            {"vllm-local": ["ollama-local"]},
        ]

    return {
        "model_list": model_list,
        "router_settings": router_settings,
    }


def start_litellm_proxy(config_path: Optional[str] = None) -> int:
    """Start LiteLLM proxy server."""
    # Generate config if not provided
    if not config_path:
        config = generate_litellm_config()
        config_path = "/tmp/litellm_config.yaml"

        # Convert to YAML format
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    # Start LiteLLM proxy
    cmd = [
        sys.executable, "-m", "litellm",
        "--config", config_path,
        "--port", str(LITELLM_PORT),
        "--host", "0.0.0.0",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    # Wait for startup
    time.sleep(3)

    if proc.poll() is not None:
        # Process exited
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"LiteLLM failed to start: {stderr}")

    # Emit bus event
    emit_bus_event(
        "local.litellm.started",
        "log",
        {"pid": proc.pid, "port": LITELLM_PORT, "config": config_path},
    )

    return proc.pid


async def infer_local(
    prompt: str,
    model: str = "vllm-local",
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> dict:
    """Run inference on local provider."""
    import aiohttp

    state = get_router_state()

    # Determine endpoint
    if model == "litellm-proxy" and state.litellm_running:
        base_url = f"http://localhost:{LITELLM_PORT}/v1"
        actual_model = "vllm-local"  # Let LiteLLM route
    elif model == "vllm-local" and state.providers.get("vllm-local", {}).get("available"):
        base_url = state.providers["vllm-local"]["base_url"]
        actual_model = state.providers["vllm-local"]["model"]
    elif model == "ollama-local" and state.providers.get("ollama-local", {}).get("available"):
        base_url = f"http://localhost:{OLLAMA_PORT}/v1"
        actual_model = state.providers["ollama-local"]["model"]
    else:
        return {"error": f"Provider {model} not available", "fallback_to": "webllm-browser"}

    # Emit request event
    req_id = f"local-{int(time.time() * 1000)}"
    emit_bus_event(
        "local.inference.request",
        "request",
        {"req_id": req_id, "model": model, "prompt_len": len(prompt)},
    )

    start = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                json={
                    "model": actual_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                latency = (time.time() - start) * 1000

                # Emit response event
                emit_bus_event(
                    "local.inference.response",
                    "response",
                    {
                        "req_id": req_id,
                        "model": model,
                        "latency_ms": latency,
                        "tokens": data.get("usage", {}).get("total_tokens", 0),
                    },
                )

                return {
                    "content": data["choices"][0]["message"]["content"],
                    "model": actual_model,
                    "latency_ms": latency,
                    "usage": data.get("usage", {}),
                }

    except Exception as e:
        emit_bus_event(
            "local.inference.error",
            "error",
            {"req_id": req_id, "model": model, "error": str(e)},
            level="error",
        )
        return {"error": str(e), "fallback_to": "webllm-browser"}


def update_vps_session_with_local_providers():
    """Update vps_session.json with local provider status."""
    state = get_router_state()

    try:
        session = {}
        if SESSION_PATH.exists():
            with open(SESSION_PATH) as f:
                session = json.load(f)

        # Update providers
        providers = session.get("providers", {})

        for name, info in state.providers.items():
            providers[name] = {
                "available": info.get("available", False),
                "last_check": info.get("checked_at", ""),
                "model": info.get("model", ""),
                "error": info.get("error"),
            }

        session["providers"] = providers

        # Update local fallback order
        session["local_fallback_order"] = state.fallback_order

        with open(SESSION_PATH, "w") as f:
            json.dump(session, f, indent=2)

    except Exception as e:
        print(f"Warning: Could not update session: {e}", file=sys.stderr)


def cmd_status(args):
    """Show status of local providers."""
    state = get_router_state()

    print("=" * 60)
    print("LOCAL LLM ROUTER STATUS")
    print("=" * 60)
    print()

    for name, info in state.providers.items():
        status = "[ONLINE]" if info.get("available") else "[OFFLINE]"
        latency = f"{info.get('latency_ms', 0):.1f}ms" if info.get("available") else "N/A"
        model = info.get("model", "")

        color = "\033[92m" if info.get("available") else "\033[91m"
        reset = "\033[0m"

        print(f"{color}{status}{reset} {name}")
        if info.get("available"):
            print(f"        Model: {model}")
            print(f"        Latency: {latency}")
            print(f"        URL: {info.get('base_url', '')}")
        else:
            print(f"        Error: {info.get('error', 'Not running')}")
        print()

    print("Fallback Order:", " -> ".join(state.fallback_order))
    print("Privacy Mode:", "ENABLED" if state.privacy_mode else "disabled")
    print()

    # Update session
    update_vps_session_with_local_providers()

    return 0


def cmd_start(args):
    """Start local router services."""
    print("Starting local LLM router...")

    state = get_router_state()

    # Check if any backends are available
    available_backends = [
        name for name, info in state.providers.items()
        if info.get("available") and name != "litellm-proxy"
    ]

    if not available_backends:
        print("Warning: No local backends (vLLM, Ollama) detected.")
        print("Start vLLM or Ollama first, or use WebLLM in browser.")
        print()
        print("To start vLLM:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model meta-llama/Llama-3.1-8B-Instruct \\")
        print("    --quantization awq --port 8000")
        print()
        print("To start Ollama:")
        print("  ollama serve &")
        print("  ollama pull llama3.2")
        return 1

    # Start LiteLLM proxy if we have backends
    try:
        pid = start_litellm_proxy()
        print(f"LiteLLM proxy started (PID: {pid}, port: {LITELLM_PORT})")

        state.litellm_running = True
        state.litellm_pid = pid
        state.started_at = datetime.utcnow().isoformat() + "Z"
        save_state(state)

        # Update session
        update_vps_session_with_local_providers()

        print()
        print("Local router ready!")
        print(f"  API endpoint: http://localhost:{LITELLM_PORT}/v1")
        print(f"  Available backends: {', '.join(available_backends)}")

        return 0

    except Exception as e:
        print(f"Error starting LiteLLM: {e}", file=sys.stderr)
        return 1


def cmd_infer(args):
    """Run inference on local provider."""
    prompt = args.prompt
    model = args.model or "vllm-local"

    print(f"Running inference on {model}...")
    print()

    result = asyncio.run(infer_local(
        prompt,
        model=model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ))

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        if result.get("fallback_to"):
            print(f"Suggested fallback: {result['fallback_to']}")
        return 1

    print("=" * 60)
    print(result["content"])
    print("=" * 60)
    print()
    print(f"Model: {result['model']}")
    print(f"Latency: {result['latency_ms']:.1f}ms")
    if result.get("usage"):
        print(f"Tokens: {result['usage']}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Local LLM Router for Pluribus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    status_parser = subparsers.add_parser("status", help="Show provider status")
    status_parser.set_defaults(func=cmd_status)

    # start
    start_parser = subparsers.add_parser("start", help="Start local router")
    start_parser.set_defaults(func=cmd_start)

    # infer
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("prompt", help="Prompt text")
    infer_parser.add_argument("--model", "-m", help="Model/provider name")
    infer_parser.add_argument("--max-tokens", type=int, default=2048)
    infer_parser.add_argument("--temperature", type=float, default=0.7)
    infer_parser.set_defaults(func=cmd_infer)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
