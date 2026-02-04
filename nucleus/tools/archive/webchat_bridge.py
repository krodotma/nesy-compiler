#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

try:
    from nucleus.tools import agent_bus
    from nucleus.tools.plurichat import execute_web_session_inference
except Exception:  # pragma: no cover
    import agent_bus  # type: ignore
    from plurichat import execute_web_session_inference  # type: ignore


DEFAULT_WEBCHAT_PROVIDERS: tuple[str, ...] = ("chatgpt-web", "claude-web", "gemini-web")


@dataclass(frozen=True)
class WebchatPaths:
    browser_state_path: str
    browser_data_dir: str
    secrets_env_path: str
    agent_home_gemini: str
    agent_home_claude: str
    agent_home_codex: str
    agent_home_gcloud: str


@dataclass(frozen=True)
class WebchatReadiness:
    daemon_running: bool
    tab_status: dict[str, str]
    missing_tabs: list[str]
    not_ready_tabs: list[str]
    browser_state_path: str
    browser_data_dir: str


@dataclass(frozen=True)
class WebchatResult:
    provider: str
    ok: bool
    latency_ms: float
    req_id: str
    response_preview: str
    error: str | None = None


def resolve_bus_dir(bus_dir: str | None = None) -> Path:
    paths = agent_bus.resolve_bus_paths(bus_dir or os.environ.get("PLURIBUS_BUS_DIR"))
    return Path(paths.active_dir)


def webchat_paths(root: Path = Path("/pluribus")) -> WebchatPaths:
    home = Path(os.environ.get("HOME") or "~").expanduser()
    secrets_env = home / ".config" / "nucleus" / "secrets.env"
    return WebchatPaths(
        browser_state_path=str(root / ".pluribus" / "browser_daemon.json"),
        browser_data_dir=str(root / ".pluribus" / "browser_data"),
        secrets_env_path=str(secrets_env),
        agent_home_gemini=str(root / ".pluribus" / "agent_homes" / "gemini"),
        agent_home_claude=str(root / ".pluribus" / "agent_homes" / "claude"),
        agent_home_codex=str(root / ".pluribus" / "agent_homes" / "codex"),
        agent_home_gcloud=str(root / ".pluribus" / "agent_homes" / "gcloud"),
    )


def webchat_readiness(
    *,
    root: Path = Path("/pluribus"),
    providers: Iterable[str] = DEFAULT_WEBCHAT_PROVIDERS,
) -> WebchatReadiness:
    state_path = root / ".pluribus" / "browser_daemon.json"
    browser_data_dir = root / ".pluribus" / "browser_data"
    tab_status: dict[str, str] = {}

    daemon_running = False
    missing_tabs: list[str] = []
    not_ready: list[str] = []

    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8", errors="replace") or "{}")
            daemon_running = bool(data.get("running"))
            tabs = data.get("tabs") if isinstance(data.get("tabs"), dict) else {}
            for p in providers:
                t = tabs.get(p) if isinstance(tabs, dict) else None
                st = ""
                if isinstance(t, dict):
                    st = str(t.get("status") or "")
                tab_status[p] = st or "missing"
        except Exception:
            daemon_running = False

    # Ensure we always return an explicit status for each requested provider.
    for p in providers:
        tab_status.setdefault(p, "missing")

    for p in providers:
        if tab_status.get(p, "missing") == "missing":
            missing_tabs.append(p)
        elif tab_status.get(p) != "ready":
            not_ready.append(p)

    return WebchatReadiness(
        daemon_running=daemon_running,
        tab_status=tab_status,
        missing_tabs=missing_tabs,
        not_ready_tabs=not_ready,
        browser_state_path=str(state_path),
        browser_data_dir=str(browser_data_dir),
    )


def send_webchat_prompt(
    *,
    prompt: str,
    provider: str,
    bus_dir: Path,
    actor: str,
    timeout_s: float = 90.0,
) -> WebchatResult:
    start = time.time()
    resp = execute_web_session_inference(prompt, provider, bus_dir, actor, timeout=timeout_s)
    latency_ms = float(getattr(resp, "latency_ms", (time.time() - start) * 1000.0) or 0.0)
    text = str(getattr(resp, "text", "") or "")
    req_id = str(getattr(resp, "req_id", "") or "")
    ok = bool(getattr(resp, "success", False))
    err = str(getattr(resp, "error", "") or "") if not ok else None
    preview = text.strip().replace("\n", " ")
    if len(preview) > 240:
        preview = preview[:239] + "…"
    return WebchatResult(
        provider=provider,
        ok=ok,
        latency_ms=latency_ms,
        req_id=req_id,
        response_preview=preview,
        error=err,
    )


def emit_bus_artifact(bus_dir: Path, *, topic: str, actor: str, data: dict) -> None:
    try:
        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind="artifact",
            level="info",
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=True,
        )
    except Exception:
        pass


def as_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def asdict_safe(obj) -> dict:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
