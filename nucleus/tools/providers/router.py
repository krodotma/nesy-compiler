#!/usr/bin/env python3
"""
Provider Router with Graceful Fallback
======================================

CORE POLICY (PluriChat): web sessions by default, no API keys.

PluriChat and the dashboard control plane route inference through persistent,
authenticated browser tabs (CUA sessions). In this profile we do NOT silently
fall back to CLI/API/mock providers.

Default profile: `web-only` (ONLY the 3 web chats, no CLI/API fallbacks).

Override (operator/dev):
- set `PLURIBUS_PROVIDER_PROFILE=full` to re-enable the legacy CLI/API mesh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from typing import IO

sys.dont_write_bytecode = True

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_loader import load_pluribus_env  # noqa: E402

try:
    from pluribus_directive import detect_pluribus_directive  # type: ignore
except Exception:  # pragma: no cover
    detect_pluribus_directive = None  # type: ignore


def emit_bus_request(topic: str, *, level: str, data: dict) -> None:
    bus_dir = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
    if not bus_dir:
        return
    tool = Path(__file__).resolve().parents[1] / "agent_bus.py"
    if not tool.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            topic,
            "--kind",
            "request",
            "--level",
            level,
            "--actor",
            os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "router",
            "--data",
            json.dumps(data, ensure_ascii=False),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

def emit_auth_required_request(*, level: str, data: dict) -> None:
    # Back-compat with older dashboards/specs that listen on the legacy topic.
    emit_bus_request("providers.web.auth.required", level=level, data=data)
    emit_bus_request("plurichat.web_session.auth.required", level=level, data=data)


def classify_blocker(text: str) -> str | None:
    t = (text or "").lower()
    if "please run /login" in t or "run /login" in t or "invalid api key" in t:
        return "auth"
    if "resource_exhausted" in t or "quota exceeded" in t or "http error: 429" in t:
        return "quota"
    if "overloaded_error" in t or "overloaded" in t or "api error: 529" in t or "http error: 529" in t:
        return "overload"
    if "no provider configured" in t or "missing api key" in t or "unsupported provider" in t:
        return "config"
    return None


def _provider_profile() -> str:
    v = (os.environ.get("PLURIBUS_PROVIDER_PROFILE") or "").strip().lower()
    # Default to web-first, no-API-key routing unless explicitly overridden.
    return v or "web-only"

def _web_only() -> bool:
    return _provider_profile() in {"web-only", "web", "plurichat", "plurichat-web", "web_session_only"}

def _allowed_providers() -> set[str] | None:
    """
    Optional provider allowlist (comma/space/newline-separated).

    If set, the router will refuse to route to any provider not in the allowlist,
    regardless of other availability checks.
    """
    raw = (os.environ.get("PLURIBUS_ALLOWED_PROVIDERS") or os.environ.get("PLURIBUS_PROVIDER_ALLOWLIST") or "").strip()
    if raw:
        parts = [p.strip().lower() for p in raw.replace("\n", ",").replace(" ", ",").split(",")]
        allowed = {p for p in parts if p}
        return allowed or None

    # Opinionated default profile: only allow the 3 web chats + Vertex Gemini.
    if _provider_profile() in {"verified", "web+vertex", "web-vertex"}:
        return {"chatgpt-web", "claude-web", "gemini-web", "vertex-gemini", "vertex-gemini-curl"}

    return None

def _filter_allowed(candidates: list[str]) -> list[str]:
    allowed = _allowed_providers()
    if not allowed:
        return candidates
    out: list[str] = []
    for c in candidates:
        cc = (c or "").strip().lower()
        if not cc:
            continue
        if cc in allowed:
            out.append(cc)
    return out

def _allow_mock() -> bool:
    return (os.environ.get("PLURIBUS_ALLOW_MOCK") or "").strip().lower() in {"1", "true", "yes", "on"}

def _require_local_only() -> bool:
    return (os.environ.get("PLURIBUS_ROUTER_REQUIRE_LOCAL") or "").strip().lower() in {"1", "true", "yes", "on"}

WEB_ONLY_PROVIDERS = ["chatgpt-web", "claude-web", "gemini-web"]
LOCAL_OVERRIDE_PROVIDERS = {
    "ollama",
    "ollama-local",
    "vllm",
    "vllm-local",
    "tensorzero",
}

_DAEMON_AUTOSTART_ATTEMPTED = False

def _autostart_browser_daemon_enabled() -> bool:
    # Default to enabled for web/verified profiles so “Providers” stays live without manual ops.
    default = "1" if _provider_profile() in {"verified", "web+vertex", "web-vertex", "web-only", "web", "plurichat", "plurichat-web", "web_session_only"} else "0"
    v = (os.environ.get("PLURIBUS_ROUTER_AUTOSTART_BROWSER_DAEMON") or default).strip().lower()
    return v in {"1", "true", "yes", "on"}

def _open_append(path: Path) -> IO[str] | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8")
    except Exception:
        return None

def _ensure_browser_daemon_running() -> None:
    """
    Best-effort: if we're in a web/verified profile and the browser daemon isn't running,
    try to start it in the background so inference can proceed without manual ops.
    """
    global _DAEMON_AUTOSTART_ATTEMPTED
    if _DAEMON_AUTOSTART_ATTEMPTED:
        return
    _DAEMON_AUTOSTART_ATTEMPTED = True

    if not _autostart_browser_daemon_enabled():
        return

    root = _find_pluribus_root()
    if not root:
        return

    state_path = root / ".pluribus" / "browser_daemon.json"
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8", errors="replace"))
            if bool(data.get("running")):
                return
        except Exception:
            pass

    daemon = Path(__file__).resolve().parents[1] / "browser_session_daemon.py"
    if not daemon.exists():
        return

    bus_dir = (os.environ.get("PLURIBUS_BUS_DIR") or str(root / ".pluribus" / "bus")).strip()
    log_file = root / ".pluribus" / "logs" / "browser_daemon_autostart.log"

    env = dict(os.environ)
    env.setdefault("PLURIBUS_BROWSER_AUTOLOGIN", "1")
    env.setdefault("PLURIBUS_BROWSER_VNC", "1")
    if env.get("PLURIBUS_BROWSER_VNC", "").strip().lower() in {"1", "true", "yes", "on"} and not env.get("DISPLAY"):
        env["DISPLAY"] = (env.get("PLURIBUS_BROWSER_DISPLAY") or ":1").strip() or ":1"

    out = _open_append(log_file)
    try:
        proc = subprocess.Popen(
            [sys.executable, str(daemon), "--root", str(root), "--bus-dir", bus_dir, "start"],
            cwd=str(root),
            env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out or subprocess.DEVNULL,
            stderr=out or subprocess.DEVNULL,
            start_new_session=True,
        )
        emit_bus_event(
            "browser.daemon.autostart",
            kind="metric",
            level="info",
            data={"ok": True, "pid": int(proc.pid), "root": str(root), "bus_dir": bus_dir},
        )
        # Give it a moment to write its state file.
        time.sleep(1.0)
    except Exception as e:
        emit_bus_event(
            "browser.daemon.autostart",
            kind="metric",
            level="warn",
            data={"ok": False, "error": str(e), "root": str(root), "bus_dir": bus_dir},
        )
    finally:
        try:
            if out:
                out.close()
        except Exception:
            pass


def have_gemini() -> bool:
    have_key = bool((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip())
    have_cli = bool(shutil.which("gemini"))
    return have_key or have_cli


def have_gemini_key() -> bool:
    return bool((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip())

def prefer_gemini_cli() -> bool:
    # Useful when an API key exists but quota/billing is not enabled.
    v = (os.environ.get("PLURIBUS_GEMINI_PREFER_CLI") or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def have_codex_cli() -> bool:
    return bool(shutil.which("codex"))


def have_claude() -> bool:
    return bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())


def have_claude_cli() -> bool:
    return bool(shutil.which("claude"))

def have_vertex_gemini() -> bool:
    have_gcloud = bool(shutil.which("gcloud"))
    have_project = bool((os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip())
    if have_gcloud and not have_project:
        # Fall back to gcloud config (common VPS posture: project set in gcloud, not env).
        base = os.environ.get("PLURIBUS_GCLOUD_HOME") or "/pluribus/.pluribus/agent_homes/gcloud"
        cfg_dir = Path(os.environ.get("CLOUDSDK_CONFIG") or str(Path(base) / ".config" / "gcloud"))
        try:
            active = (cfg_dir / "active_config").read_text(encoding="utf-8", errors="replace").strip() or "default"
        except Exception:
            active = "default"
        cfg_file = cfg_dir / "configurations" / f"config_{active}"
        if cfg_file.exists():
            try:
                for line in cfg_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.strip().startswith("project") and "=" in line:
                        val = line.split("=", 1)[1].strip()
                        if val:
                            have_project = True
                            break
            except Exception:
                pass
    return have_gcloud and have_project


def have_tensorzero() -> bool:
    """Check if TensorZero gateway is configured."""
    return bool((os.environ.get("TENSORZERO_GATEWAY_URL") or "").strip())

def have_vllm() -> bool:
    """Check if a local vLLM (OpenAI-compatible) server is reachable."""
    return _vllm_ready()

def have_ollama() -> bool:
    """Check if a local Ollama server is reachable."""
    try:
        import urllib.request

        req = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.5)
        return req.status == 200
    except Exception:
        return False


def have_grok() -> bool:
    """Check if Grok/xAI API key is available."""
    return bool((os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or "").strip())


def have_grok_cli() -> bool:
    """Check if grok CLI is available (vibe-kit or official)."""
    return bool(shutil.which("grok"))

def _find_pluribus_root() -> Path | None:
    # Prefer global /pluribus, else search upward from CWD.
    cand = Path("/pluribus")
    if (cand / ".pluribus" / "rhizome.json").exists():
        return cand
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / ".pluribus" / "rhizome.json").exists():
            return p
    return None


def _vps_active_fallback() -> str | None:
    root = _find_pluribus_root()
    if not root:
        return None
    path = root / ".pluribus" / "vps_session.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    v = obj.get("active_fallback")
    return str(v) if isinstance(v, str) and v else None


def _vps_fallback_order() -> list[str] | None:
    root = _find_pluribus_root()
    if not root:
        return None
    path = root / ".pluribus" / "vps_session.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    v = obj.get("fallback_order")
    if not isinstance(v, list):
        return None
    out: list[str] = []
    for item in v:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out or None


def _map_fallback_to_provider(fb: str) -> str | None:
    m = {
        # Browser-backed providers (existing logged-in web sessions).
        "chatgpt-web": "chatgpt-web",
        "gemini-web": "gemini-web",
        "claude-web": "claude-web",
        "codex-cli": "codex-cli",
        "gemini": "gemini",
        "gemini-cli": "gemini-cli",
        "vertex-gemini": "vertex-gemini",
        "vertex-gemini-curl": "vertex-gemini-curl",
        "claude-api": "claude-api",
        "claude-cli": "claude-cli",
        "mock": "mock",
        # Local providers
        "vllm-local": "vllm-local",
        "vllm": "vllm-local",
        "ollama-local": "ollama-local",
        "ollama": "ollama-local",
        "tensorzero": "tensorzero",
        # Grok/xAI providers
        "grok": "grok-cli",
        "grok-cli": "grok-cli",
        "xai": "grok-cli",
    }
    return m.get((fb or "").strip())

def _browser_tab_ready(provider_id: str) -> bool:
    root = _find_pluribus_root()
    if not root:
        return False
    state_path = root / ".pluribus" / "browser_daemon.json"
    if not state_path.exists():
        return False
    try:
        data = json.loads(state_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    if not bool(data.get("running")):
        return False
    tabs = data.get("tabs")
    if not isinstance(tabs, dict):
        return False
    tab = tabs.get(provider_id)
    if not isinstance(tab, dict):
        return False
    return str(tab.get("status") or "").strip().lower() == "ready"

def _vllm_ready() -> bool:
    try:
        # Check standard port 8000 (OpenAI compatible)
        import urllib.request
        req = urllib.request.urlopen("http://localhost:8000/health", timeout=0.5)
        return req.status == 200
    except Exception:
        return False

def pick_provider(requested: str, *, model: str | None = None) -> str:
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        allowed = _allowed_providers()
        if allowed and requested not in allowed:
            if requested == "mock" and _allow_mock() and "mock" in allowed:
                return "mock"
            return "none"
        # In web-only mode, reject non-web providers even when explicitly requested.
        if _web_only() and requested not in set(WEB_ONLY_PROVIDERS):
            if requested == "mock" and _allow_mock():
                return "mock"
            return "none"
        return requested

    model_norm = (model or "").strip().lower()
    # Control-plane override: if VPS session has an active fallback, honor it.
    # This lets TUI/Web switch providers without changing local env vars.
    active_fb = _vps_active_fallback()
    if active_fb:
        mapped = _map_fallback_to_provider(active_fb)
        if _web_only() and mapped and mapped not in set(WEB_ONLY_PROVIDERS):
            mapped = None
        # If the caller requires Gemini-3, prefer Vertex even if codex is "available".
        if model_norm.startswith("gemini-3") and mapped and not mapped.startswith("vertex-gemini"):
            if have_vertex_gemini():
                return "vertex-gemini-curl" if shutil.which("curl") else "vertex-gemini"
        if mapped:
            return mapped

    # Model-aware routing: if the caller requested an explicit model family, prefer a provider
    # that can actually satisfy it (before generic speed-first fallbacks like vLLM).
    if model_norm:
        if model_norm.startswith("gemini-3") and not _web_only() and have_vertex_gemini():
            return "vertex-gemini-curl" if shutil.which("curl") else "vertex-gemini"
        if "gemini" in model_norm:
            if _browser_tab_ready("gemini-web"):
                return "gemini-web"
            if _web_only():
                return "none"
            if have_vertex_gemini():
                return "vertex-gemini-curl" if shutil.which("curl") else "vertex-gemini"
            if have_gemini():
                return "gemini"
        if "claude" in model_norm:
            if _browser_tab_ready("claude-web"):
                return "claude-web"
            if _web_only():
                return "none"
            if have_claude():
                return "claude-api"
            if have_claude_cli():
                return "claude-cli"
        if "gpt" in model_norm or "openai" in model_norm:
            if _browser_tab_ready("chatgpt-web"):
                return "chatgpt-web"
            if _web_only():
                return "none"
            if have_codex_cli():
                return "codex-cli"
        if "grok" in model_norm or "xai" in model_norm:
            if _web_only():
                return "none"
            if have_grok() or have_grok_cli():
                return "grok-cli"

    # Priority 1: Local vLLM (Elite Speed)
    if not _web_only() and _vllm_ready():
        return "vllm-local"

    # Priority 2: Web Sessions
    if _web_only():
        preferred = WEB_ONLY_PROVIDERS
        if model_norm:
            if "claude" in model_norm:
                preferred = ["claude-web", "chatgpt-web", "gemini-web"]
            elif "gemini" in model_norm:
                preferred = ["gemini-web", "chatgpt-web", "claude-web"]
            elif "gpt" in model_norm or "openai" in model_norm:
                preferred = ["chatgpt-web", "claude-web", "gemini-web"]
        for pid in preferred:
            if _browser_tab_ready(pid):
                return pid
        return "none"

    preferred = WEB_ONLY_PROVIDERS
    if model_norm:
        if "claude" in model_norm:
            preferred = ["claude-web", "chatgpt-web", "gemini-web"]
        elif "gemini" in model_norm:
            preferred = ["gemini-web", "chatgpt-web", "claude-web"]
        elif "gpt" in model_norm or "openai" in model_norm:
            preferred = ["chatgpt-web", "claude-web", "gemini-web"]
    for pid in preferred:
        if _browser_tab_ready(pid):
            return pid
    if have_tensorzero():
        return "tensorzero"
    if have_codex_cli():
        return "codex-cli"
    if have_gemini():
        return "gemini"
    if have_vertex_gemini():
        return "vertex-gemini"
    if have_claude():
        return "claude-api"
    if have_claude_cli():
        return "claude-cli"
    return "none"


def run(argv: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(p.returncode), p.stdout, p.stderr


def _safe_excerpt(text: str, *, max_chars: int = 500) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "…"


def emit_bus_event(topic: str, *, kind: str, level: str, data: dict) -> None:
    bus_dir = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
    if not bus_dir:
        return
    tool = Path(__file__).resolve().parents[1] / "agent_bus.py"
    if not tool.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            topic,
            "--kind",
            kind,
            "--level",
            level,
            "--actor",
            os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "router",
            "--data",
            json.dumps(data, ensure_ascii=False),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def emit_incident(provider: str, *, blocker: str | None, code: int, err: str, out: str) -> None:
    # Minimal, non-secret, replay-friendly incident record.
    cooldown_s = 0
    if blocker == "overload":
        cooldown_s = 120
    data = {
        "provider": provider,
        "available": False,
        "blocker": blocker,
        "exit_code": int(code),
        "error": _safe_excerpt(err),
        "stdout": _safe_excerpt(out),
        "cooldown_s": cooldown_s,
    }
    emit_bus_event("providers.incident", kind="metric", level="warn", data=data)
    if provider.startswith("claude") and blocker == "overload":
        # Keep legacy topic for existing dashboards/alerts.
        emit_bus_event("providers.claude.overloaded", kind="metric", level="warn", data={"status": "overloaded", "http": 529, "message": "overloaded", "provider": provider})


def build_cmd(provider: str, *, prompt: str, model: str | None) -> tuple[str, list[str]] | None:
    tool_dir = os.path.dirname(__file__)
    p = (provider or "").strip().lower()
    tier = os.environ.get("PLURIBUS_TIER", "prod")
    
    if p in {"chatgpt-web", "gemini-web", "claude-web"}:
        root = _find_pluribus_root() or Path("/pluribus")
        bus_dir = (os.environ.get("PLURIBUS_BUS_DIR") or str(root / ".pluribus" / "bus")).strip()
        daemon = Path(__file__).resolve().parents[1] / "browser_session_daemon.py"
        cmd = [
            sys.executable,
            str(daemon),
            "--root",
            str(root),
            "--bus-dir",
            bus_dir,
            "infer",
            p,
            prompt,
        ]
    elif p in {"codex", "codex-cli"}:
        cmd = [sys.executable, os.path.join(tool_dir, "codex_cli_smoke.py"), "--prompt", prompt]
    elif p == "gemini":
        # Prefer API key (REST); fall back to gemini CLI (web auth) if installed.
        if have_gemini_key() and not (prefer_gemini_cli() and shutil.which("gemini")):
            cmd = [sys.executable, os.path.join(tool_dir, "gemini_smoke.py"), "--prompt", prompt]
        else:
            cmd = [sys.executable, os.path.join(tool_dir, "gemini_cli_smoke.py"), "--prompt", prompt]
    elif p == "gemini-cli":
        cmd = [sys.executable, os.path.join(tool_dir, "gemini_cli_smoke.py"), "--prompt", prompt]
    elif p == "vertex-gemini":
        cmd = [sys.executable, os.path.join(tool_dir, "vertex_gemini_smoke.py"), "--prompt", prompt]
    elif p == "vertex-gemini-curl":
        cmd = [sys.executable, os.path.join(tool_dir, "vertex_gemini_curl_smoke.py"), "--prompt", prompt]
    elif p in {"claude", "claude-api"}:
        # Back-compat: "claude" means API if key is present, else CLI if available.
        if not have_claude() and have_claude_cli():
            p = "claude-cli"
            claude_home = os.environ.get("PLURIBUS_CLAUDE_HOME") or "/pluribus/.pluribus/agent_homes/claude"
            try:
                os.makedirs(claude_home, exist_ok=True)
            except Exception:
                pass
            cmd = [
                sys.executable,
                os.path.join(tool_dir, "claude_cli_smoke.py"),
                "--prompt",
                prompt,
                "--home",
                claude_home,
            ]
        else:
            cmd = [sys.executable, os.path.join(tool_dir, "claude_smoke.py"), "--prompt", prompt]
    elif p == "claude-cli":
        claude_home = os.environ.get("PLURIBUS_CLAUDE_HOME") or "/pluribus/.pluribus/agent_homes/claude"
        try:
            os.makedirs(claude_home, exist_ok=True)
        except Exception:
            pass
        cmd = [
            sys.executable,
            os.path.join(tool_dir, "claude_cli_smoke.py"),
            "--prompt",
            prompt,
            "--home",
            claude_home,
        ]
    elif p == "mock":
        cmd = [sys.executable, os.path.join(tool_dir, "mock_smoke.py"), "--prompt", prompt]
    elif p in {"vllm", "vllm-local"}:
        p = "vllm-local"
        cmd = [sys.executable, os.path.join(tool_dir, "vllm_smoke.py"), "--prompt", prompt]
    elif p in {"ollama", "ollama-local"}:
        p = "ollama-local"
        cmd = [sys.executable, os.path.join(tool_dir, "ollama_smoke.py"), "--prompt", prompt]
    elif p == "tensorzero":
        cmd = [sys.executable, os.path.join(tool_dir, "tensorzero_smoke.py"), "--prompt", prompt]
    elif p in {"grok", "grok-cli", "xai"}:
        p = "grok-cli"
        # Grok smoke is in pluribus_next/tools/providers/
        grok_smoke = Path(__file__).resolve().parents[2] / "pluribus_next" / "tools" / "providers" / "grok_cli_smoke.py"
        if not grok_smoke.exists():
            # Fallback: same directory
            grok_smoke = Path(tool_dir) / "grok_cli_smoke.py"
        cmd = [sys.executable, str(grok_smoke), "--prompt", prompt]
    else:
        return None
    if model and p not in {"chatgpt-web", "gemini-web", "claude-web"}:
        cmd += ["--model", model]
    return p, cmd


def candidate_providers(requested: str, *, model: str | None) -> list[str]:
    # Keep browser daemon alive for autonomy: start it if needed (web/verified profiles).
    _ensure_browser_daemon_running()
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        allowed = _allowed_providers()
        if allowed and requested not in allowed:
            if requested == "mock" and _allow_mock() and "mock" in allowed:
                return ["mock"]
            return []
        if _web_only():
            if requested == "mock":
                return ["mock"] if _allow_mock() else []
            if requested in set(WEB_ONLY_PROVIDERS):
                return [requested] if _browser_tab_ready(requested) else []
            # Allow explicit local overrides without API keys.
            if requested in LOCAL_OVERRIDE_PROVIDERS:
                return _filter_allowed([requested])
            return []
        return _filter_allowed([requested])

    allow_mock = _allow_mock()
    if _require_local_only():
        # Security posture: never route to web/cloud providers for this request.
        # Intended to be set by a higher-level policy engine (e.g., PBVW).
        out: list[str] = []
        if have_vllm():
            out.append("vllm-local")
        if have_ollama():
            out.append("ollama-local")
        if allow_mock:
            out.append("mock")
        return _filter_allowed(out)

    picked = pick_provider("auto", model=model)
    order = _vps_fallback_order() or []
    mapped = [(_map_fallback_to_provider(x) or "").strip() for x in order]
    candidates = [picked, *mapped]
    if _web_only():
        # Only include ready web providers. If none are ready, return [] so callers
        # can surface an actionable login/daemon error instead of trying CLI/API fallbacks.
        out: list[str] = []
        for c in [*candidates, *WEB_ONLY_PROVIDERS]:
            c = (c or "").strip().lower()
            if not c or c == "none":
                continue
            if c not in set(WEB_ONLY_PROVIDERS):
                continue
            if not _browser_tab_ready(c):
                continue
            if c not in out:
                out.append(c)
        return _filter_allowed(out)
    # Heuristic tail: availability checks (keeps behavior when vps session is missing)
    # Local providers first (fastest, no API keys), then cloud fallbacks
    if have_tensorzero():
        candidates.append("tensorzero")
    if have_vllm():
        candidates.append("vllm-local")
    if have_ollama():
        candidates.append("ollama-local")
    # Prefer browser-backed sessions first (when available), then CLIs/APIs.
    candidates += ["chatgpt-web", "claude-web", "gemini-web", "codex-cli", "gemini", "vertex-gemini", "claude-api", "claude-cli"]
    # Add Grok if available (cheap coding model option).
    if have_grok() or have_grok_cli():
        candidates.append("grok-cli")
    if allow_mock:
        candidates.append("mock")

    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        c = (c or "").strip().lower()
        if not c or c == "none":
            continue
        if c == "mock" and not allow_mock:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return _filter_allowed(out)


def main(argv: list[str]) -> int:
    load_pluribus_env()
    p = argparse.ArgumentParser(prog="router.py", description="Provider router (Gemini/Claude) using existing smoke tools.")
    p.add_argument(
        "--provider",
        default="auto",
        help="auto|chatgpt-web|gemini-web|claude-web|vllm-local|ollama-local|codex|codex-cli|gemini|gemini-cli|vertex-gemini|vertex-gemini-curl|claude|claude-api|claude-cli|grok|grok-cli|mock",
    )
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default=None, help="Optional model override (provider-specific env vars still apply).")
    p.add_argument("--format", default="text", choices=["text", "json"], help="Output format for stdout (text|json).")
    args = p.parse_args(argv)
    out_json = str(args.format or "text").strip().lower() == "json"

    pluribus_directive = None
    if detect_pluribus_directive is not None:
        try:
            pluribus_directive = detect_pluribus_directive(str(args.prompt or ""))
        except Exception:
            pluribus_directive = None
    if pluribus_directive is not None:
        req_id = (os.environ.get("PLURIBUS_GATEWAY_REQ_ID") or os.environ.get("PLURIBUS_REQ_ID") or "").strip() or str(uuid.uuid4())
        prompt_sha256 = hashlib.sha256(str(args.prompt or "").encode("utf-8", errors="replace")).hexdigest()
        emit_bus_event(
            "pluribus.directive.detected",
            kind="artifact",
            level="info",
            data={
                "req_id": req_id,
                "source": "providers.router",
                "provider_arg": str(args.provider or "auto"),
                "model": str(args.model or ""),
                "prompt_sha256": prompt_sha256,
                "directive": pluribus_directive.to_bus_dict(),  # type: ignore[union-attr]
            },
        )

    providers = _filter_allowed(candidate_providers(args.provider, model=args.model))
    if not providers:
        if _web_only() or (_allowed_providers() and {"chatgpt-web", "claude-web", "gemini-web"} & (_allowed_providers() or set())):
            # Non-blocking: request human-in-loop authentication via VNC (dashboard).
            emit_auth_required_request(
                level="warn",
                data={
                    "status": "blocked",
                    "reason": "no_web_session_ready",
                    "providers": [p for p in WEB_ONLY_PROVIDERS if not _allowed_providers() or p in (_allowed_providers() or set())],
                    "vnc_url": (os.environ.get("PLURIBUS_DASHBOARD_URL") or "https://kroma.live") + "/vnc/vnc.html",
                    "browser_auth_url": (os.environ.get("PLURIBUS_DASHBOARD_URL") or "https://kroma.live"),
                    "req_id": (os.environ.get("PLURIBUS_GATEWAY_REQ_ID") or os.environ.get("PLURIBUS_REQ_ID") or "").strip() or None,
                },
            )
        msg = (
            "no web session ready (start browser_session_daemon and ensure chatgpt/claude/gemini tabs are logged in)"
            if _web_only()
            else "no provider configured (set GEMINI_API_KEY/GOOGLE_API_KEY, ANTHROPIC_API_KEY, or install claude CLI)"
        )
        if out_json:
            sys.stdout.write(
                json.dumps(
                    {
                        "ok": False,
                        "provider": "none",
                        "model": args.model,
                        "exit_code": 2,
                        "error": msg,
                        "ts": int(time.time()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            return 2
        sys.stderr.write(msg + "\n")
        return 2

    def _web_auth_indicated(err_text: str, out_text: str) -> bool:
        t = f"{err_text or ''}\n{out_text or ''}".lower()
        return any(
            s in t
            for s in [
                "daemon not running",
                "needs_login",
                "needs_onboarding",
                "needs_code",
                "blocked_bot",
                "auth required",
                "no creds in env",
                "no web session ready",
            ]
        )

    emitted_web_auth_request = False
    last_code, last_out, last_err = 2, "", "unsupported provider"
    last_provider = "none"
    for idx, provider in enumerate(providers):
        built = build_cmd(provider, prompt=args.prompt, model=args.model)
        if not built:
            last_code, last_out, last_err = 2, "", f"unsupported provider: {provider}"
            continue
        norm, cmd = built
        last_provider = norm
        code, out, err = run(cmd)

        # If a web-session provider fails due to auth/daemon state, emit a non-blocking request
        # so the dashboard can prompt for VNC-based login/2FA.
        if not emitted_web_auth_request and norm in set(WEB_ONLY_PROVIDERS) and code != 0 and _web_auth_indicated(err, out):
            emitted_web_auth_request = True
            emit_auth_required_request(
                level="warn",
                data={
                    "status": "blocked",
                    "reason": "web_auth_required",
                    "providers": [p for p in WEB_ONLY_PROVIDERS if not _allowed_providers() or p in (_allowed_providers() or set())],
                    "vnc_url": (os.environ.get("PLURIBUS_DASHBOARD_URL") or "https://kroma.live") + "/vnc/vnc.html",
                    "browser_auth_url": (os.environ.get("PLURIBUS_DASHBOARD_URL") or "https://kroma.live"),
                    "req_id": (os.environ.get("PLURIBUS_GATEWAY_REQ_ID") or os.environ.get("PLURIBUS_REQ_ID") or "").strip() or None,
                },
            )

        # Gemini free-tier keys can be present but unusable (quota=0); fall back to CLI if available.
        if norm == "gemini" and code != 0 and (("RESOURCE_EXHAUSTED" in (err or "")) or ("Quota exceeded" in (err or "")) or ("http error: 429" in (err or ""))):
            if shutil.which("gemini"):
                built2 = build_cmd("gemini-cli", prompt=args.prompt, model=args.model)
                if built2:
                    norm2, cmd2 = built2
                    code, out, err = run(cmd2)
                    norm = norm2

        # Claude API keys might be invalid/out-of-credits/overloaded; fallback to CLI (web auth) if available.
        if norm in {"claude", "claude-api"} and code != 0:
            err_lower = (err or "").lower()
            if "invalid x-api-key" in err_lower or "401" in err_lower or "403" in err_lower or "credit balance is too low" in err_lower or "overloaded" in err_lower:
                if shutil.which("claude"):
                    built2 = build_cmd("claude-cli", prompt=args.prompt, model=args.model)
                    if built2:
                        norm2, cmd2 = built2
                        code, out, err = run(cmd2)
                        norm = norm2

        if code == 0:
            if out_json:
                sys.stdout.write(
                    json.dumps(
                        {
                            "ok": True,
                            "provider": norm,
                            "model": args.model,
                            "text": out or "",
                            "stderr": err or "",
                            "ts": int(time.time()),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                return 0
            if err:
                sys.stderr.write(err)
            if out:
                sys.stdout.write(out)
            return 0

        blocker = classify_blocker("\n".join([err or "", out or ""]))
        emit_incident(norm, blocker=blocker, code=code, err=err or "", out=out or "")

        if blocker == "auth" and norm.startswith("claude"):
            emit_bus_request(
                "providers.claude.auth.required",
                level="warn",
                data={
                    "status": "blocked",
                    "reason": "claude-cli needs /login",
                    "next": "HOME=/pluribus/.pluribus/agent_homes/claude /pluribus/nucleus/tools/providers/claude_setup_token.sh",
                },
            )
        elif blocker == "quota" and norm.startswith("gemini"):
            emit_bus_request("providers.gemini.quota.blocked", level="warn", data={"status": "blocked", "reason": "gemini quota exhausted"})
        elif blocker == "config":
            emit_bus_request("providers.config.required", level="warn", data={"status": "blocked", "reason": "provider config missing"})

        last_code, last_out, last_err = code, out, err
        # Only iterate across providers when the caller opted into auto routing.
        if (args.provider or "auto").strip().lower() != "auto":
            break

    if last_err:
        if not out_json:
            sys.stderr.write(last_err)
    if last_out:
        if not out_json:
            sys.stdout.write(last_out)
    if out_json:
        msg = (last_err or last_out or "router error").strip()
        sys.stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "provider": last_provider,
                    "model": args.model,
                    "exit_code": int(last_code),
                    "error": msg,
                    "stdout": last_out or "",
                    "stderr": last_err or "",
                    "ts": int(time.time()),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return int(last_code)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
