#!/usr/bin/env python3
"""VPS Session Manager.

Tracks provider availability, fallback modes, and PBPAIR state.
Mirrors the TypeScript VPSSession interface for isomorphic compatibility.

Usage:
    python3 vps_session.py status     # Show current session status
    python3 vps_session.py refresh    # Refresh provider availability
    python3 vps_session.py mode m|A   # Set flow mode
    python3 vps_session.py daemon    # Bus-controlled refresh/mode + periodic refresh loop
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    # Atomic write to avoid partially-written JSON being observed by readers (e.g. bus-bridge /session).
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None

def _read_tail_bytes(path: Path, *, max_bytes: int) -> bytes:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            start = max(0, end - max_bytes)
            f.seek(start)
            return f.read(end - start)
    except Exception:
        return b""


@dataclass
class ProviderStatus:
    """Status of a provider."""
    available: bool = False
    last_check: str = ""
    error: str | None = None
    quota_remaining: int | None = None
    model: str | None = None
    browser: str | None = None


@dataclass
class PBPAIRRequest:
    """PBPAIR request state."""
    id: str = ""
    provider: str = ""
    role: str = ""
    prompt: str = ""
    flow_mode: str = "m"
    created_iso: str = ""
    status: str = "pending"  # pending, proposed, approved, completed, rejected


@dataclass
class VPSSession:
    """VPS Session state - mirrors TypeScript interface."""
    flow_mode: str = "m"  # m (monitor) or A (automatic)

    # Provider status
    providers: dict[str, ProviderStatus] = field(default_factory=lambda: {
        "chatgpt-web": ProviderStatus(),
        "claude-web": ProviderStatus(),
        "gemini-web": ProviderStatus(),
        "ollama-local": ProviderStatus(),
        "vllm-local": ProviderStatus(),
    })

    fallback_order: list[str] = field(default_factory=lambda: [
        "chatgpt-web",
        "claude-web",
        "gemini-web",
        "ollama-local",
        "vllm-local",
    ])
    active_fallback: str | None = None
    # Temporary provider cooldowns (unix ts until which provider is treated unavailable).
    # Used to handle transient outages/overload without blocking the mesh.
    provider_cooldowns: dict[str, float] = field(default_factory=dict)

    # PBPAIR state
    pbpair_requests: list[PBPAIRRequest] = field(default_factory=list)

    # Auth state
    gcp_project: str | None = None
    gcp_location: str | None = None
    claude_logged_in: bool = False
    gemini_cli_logged_in: bool = False
    openai_web_logged_in: bool = False


class VPSSessionManager:
    """Manages VPS session state."""

    def __init__(self, root: Path):
        self.root = root
        self.pluribus_dir = root / ".pluribus"
        self.session_path = self.pluribus_dir / "vps_session.json"
        self.tools_dir = root / "nucleus" / "tools"
        self._session: VPSSession | None = None

    def load(self) -> VPSSession:
        """Load session from disk."""
        data = read_json(self.session_path)

        if data:
            session = VPSSession(
                flow_mode=data.get("flow_mode", "m"),
                fallback_order=data.get("fallback_order", VPSSession().fallback_order),
                active_fallback=data.get("active_fallback"),
                provider_cooldowns={},
                gcp_project=data.get("gcp_project"),
                gcp_location=data.get("gcp_location"),
                claude_logged_in=data.get("claude_logged_in", False),
                gemini_cli_logged_in=data.get("gemini_cli_logged_in", False),
            )

            # Load provider cooldowns (optional)
            cds = data.get("provider_cooldowns")
            if isinstance(cds, dict):
                for k, v in cds.items():
                    try:
                        session.provider_cooldowns[str(k)] = float(v)
                    except Exception:
                        continue

            # Load provider status
            for name, status_data in data.get("providers", {}).items():
                if name in session.providers:
                    session.providers[name] = ProviderStatus(
                        available=status_data.get("available", False),
                        last_check=status_data.get("last_check", ""),
                        error=status_data.get("error"),
                        quota_remaining=status_data.get("quota_remaining"),
                        model=status_data.get("model"),
                    )

            self._session = session
        else:
            self._session = VPSSession()

        # Read environment overrides
        if os.environ.get("PLURIBUS_FLOW_MODE"):
            self._session.flow_mode = os.environ["PLURIBUS_FLOW_MODE"]

        fallback_env = os.environ.get("PLURIBUS_FALLBACK_PROVIDERS")
        if fallback_env:
            self._session.fallback_order = fallback_env.split(",")

        if os.environ.get("VERTEX_PROJECT"):
            self._session.gcp_project = os.environ["VERTEX_PROJECT"]

        if os.environ.get("VERTEX_LOCATION"):
            self._session.gcp_location = os.environ["VERTEX_LOCATION"]

        # Enforce "mock is internal-only" unless explicitly allowed.
        allow_mock = (os.environ.get("PLURIBUS_ALLOW_MOCK") or "").strip().lower() in {"1", "true", "yes", "on"}
        if not allow_mock:
            self._session.fallback_order = [p for p in self._session.fallback_order if str(p).strip() and str(p).strip() != "mock"]
            if "mock" in self._session.providers:
                self._session.providers["mock"].available = False
                self._session.providers["mock"].error = self._session.providers["mock"].error or "internal-only"
            if self._session.active_fallback == "mock":
                self._session.active_fallback = None

        # Default profile is "verified": keep ONLY web sessions + Vertex Gemini in the chain so the UI
        # doesn't surface stale CLI/API options from older session files.
        profile = (os.environ.get("PLURIBUS_PROVIDER_PROFILE") or "").strip().lower() or "verified"
        web_only = profile in {"web-only", "web", "plurichat", "plurichat-web", "web_session_only"}
        verified = profile in {"verified", "web+vertex", "web-vertex"}
        if web_only or verified:
            allowed = {"chatgpt-web", "claude-web", "gemini-web", "ollama-local", "vllm-local"} | ({"mock"} if allow_mock else set())
            if verified:
                allowed |= {"vertex-gemini", "vertex-gemini-curl"}
            self._session.fallback_order = [p for p in self._session.fallback_order if str(p).strip() in allowed]
            if self._session.active_fallback and str(self._session.active_fallback).strip() not in allowed:
                self._session.active_fallback = None
            # Mark disallowed providers as unavailable for UI clarity.
            for pid, st in list(self._session.providers.items()):
                if str(pid).strip() in allowed:
                    continue
                st.available = False
                st.error = st.error or "disabled by provider profile"
        # Ensure web-session providers + local are present early in the chain (control-plane default),
        # unless the operator explicitly overrides the chain via env.
        preferred = ["chatgpt-web", "claude-web", "gemini-web", "ollama-local", "vllm-local"] if not fallback_env else []
        seen: set[str] = set()
        normalized: list[str] = []
        for p in [*preferred, *self._session.fallback_order]:
            pid = str(p).strip()
            if not pid:
                continue
            if pid == "mock" and not allow_mock:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            normalized.append(pid)
        self._session.fallback_order = normalized

        return self._session

    def save(self) -> None:
        """Save session to disk."""
        if not self._session:
            return

        data = {
            "flow_mode": self._session.flow_mode,
            "providers": {
                name: asdict(status)
                for name, status in self._session.providers.items()
            },
            "fallback_order": self._session.fallback_order,
            "active_fallback": self._session.active_fallback,
            "provider_cooldowns": self._session.provider_cooldowns,
            "gcp_project": self._session.gcp_project,
            "gcp_location": self._session.gcp_location,
            "claude_logged_in": self._session.claude_logged_in,
            "gemini_cli_logged_in": self._session.gemini_cli_logged_in,
            "updated_iso": now_iso_utc(),
        }

        write_json(self.session_path, data)

    def set_flow_mode(self, mode: str) -> None:
        """Set flow mode (m or A)."""
        if mode not in ("m", "A"):
            raise ValueError(f"Invalid flow mode: {mode}. Must be 'm' or 'A'")

        session = self.load()
        session.flow_mode = mode
        self.save()

        # Emit bus event
        self._emit_bus("dashboard.vps.flow_mode_changed", {"mode": mode})

    def refresh_provider(self, name: str) -> ProviderStatus:
        """Refresh a specific provider's status."""
        session = self.load()

        # Local providers via local_router logic
        if name in ("ollama-local", "vllm-local"):
            try:
                # Add current directory to path to find local_router
                if str(self.tools_dir) not in sys.path:
                    sys.path.append(str(self.tools_dir))
                import local_router
                if name == "ollama-local":
                    lp = local_router.check_ollama()
                else:
                    lp = local_router.check_vllm()
                status = ProviderStatus(
                    available=lp.available,
                    last_check=lp.checked_at,
                    model=lp.model,
                    error=lp.error,
                    browser="native"
                )
            except Exception as e:
                status = ProviderStatus(available=False, last_check=now_iso_utc(), error=str(e), browser="native")
            
            old_status = session.providers.get(name)
            session.providers[name] = status
            self.save()
            
            # Emit bus event only if status changed
            if not old_status or old_status.available != status.available or old_status.error != status.error:
                self._emit_bus("dashboard.vps.provider_status", {"provider": name, "available": status.available, "error": status.error, "browser": status.browser})
            
            return status

        # Web sessions are provided by the browser_session_daemon (best-effort; can be blocked by bot checks).
        if name in ("gemini-web", "claude-web", "chatgpt-web"):
            state_path = self.pluribus_dir / "browser_daemon.json"
            status = ProviderStatus(available=False, last_check=now_iso_utc(), error="browser daemon not running")
            try:
                data = read_json(state_path) or {}
                pid = int(data.get("pid") or 0)
                running = bool(data.get("running")) and pid > 0
                if running:
                    try:
                        os.kill(pid, 0)
                    except Exception:
                        running = False
                tabs = data.get("tabs") or {}
                tab = tabs.get(name) or {}
                t_status = str(tab.get("status") or "")
                t_err = tab.get("error")
                
                # Try to determine browser type (default to chromium)
                browser_type = "chromium"
                if name == "gemini-web":
                    browser_type = "firefox"

                if not running:
                    status.available = False
                    status.error = "browser daemon not running"
                    status.browser = browser_type
                elif t_status == "ready":
                    status.available = True
                    status.error = None
                    status.browser = browser_type
                else:
                    status.available = False
                    status.error = (t_err or t_status or "unavailable")
                    status.browser = browser_type
            except Exception as e:
                status.available = False
                status.error = str(e)[:200]
            # Only emit event if status changed
            old_status = session.providers.get(name)
            session.providers[name] = status
            self.save()

            # Emit bus event only if status changed
            if not old_status or old_status.available != status.available or old_status.error != status.error:
                self._emit_bus("dashboard.vps.provider_status", {"provider": name, "available": status.available, "error": status.error, "browser": status.browser})

            return status

        # Map provider names to fast auth check scripts (no LLM calls, just CLI presence + auth)
        auth_check_scripts = {
            "gemini": "providers/gemini_auth_check.py",
            "claude": "providers/claude_auth_check.py",
            "codex": "providers/codex_auth_check.py",
            "vertex": "providers/vertex_auth_check.py",
            "vertex-curl": "providers/vertex_auth_check.py",
            "openai": "providers/openai_auth_check.py",
            "github": "providers/github_auth_check.py",
            "mock": "providers/mock_smoke.py",
        }

        script = auth_check_scripts.get(name)
        if not script:
            return session.providers.get(name, ProviderStatus())

        script_path = self.tools_dir / script
        if not script_path.exists():
            session.providers[name] = ProviderStatus(
                available=False,
                last_check=now_iso_utc(),
                error=f"Auth check not found: {script}",
            )
            self.save()
            return session.providers[name]

        # Some checks need extra flags, but no LLM calls should happen here.
        cmd = [sys.executable, str(script_path)]
        if name == "vertex-curl":
            cmd += ["--require-curl"]
        if script_path.name.endswith("_smoke.py"):
            cmd += ["--prompt", "Say OK"]

        try:
            # Keep the outer timeout >= provider script internal timeouts to avoid false "Timeout".
            # Claude/Gemini CLIs can take 15-20s on first invocation due to startup overhead.
            timeout_s_map = {
                "gemini": 40,
                "claude": 40,
                "codex": 10,
                "openai": 10,
                "github": 15,
                "vertex": 70,
                "vertex-curl": 70,
                "mock": 10,
            }
            timeout_s = int(timeout_s_map.get(name, 15))

            # Ensure provider checks run against the shared agent homes (no API keys required).
            env = dict(os.environ)
            env.setdefault("PLURIBUS_CODEX_HOME", "/pluribus/.pluribus/agent_homes/codex")
            env.setdefault("PLURIBUS_CLAUDE_HOME", "/pluribus/.pluribus/agent_homes/claude")
            env.setdefault("PLURIBUS_GEMINI_HOME", "/pluribus/.pluribus/agent_homes/gemini")
            env.setdefault("PLURIBUS_GCLOUD_HOME", "/pluribus/.pluribus/agent_homes/gcloud")
            if name == "codex":
                env["HOME"] = env["PLURIBUS_CODEX_HOME"]
            elif name == "claude":
                env["HOME"] = env["PLURIBUS_CLAUDE_HOME"]
            elif name == "gemini":
                env["HOME"] = env["PLURIBUS_GEMINI_HOME"]
            elif name.startswith("vertex"):
                env["HOME"] = env["PLURIBUS_GCLOUD_HOME"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(self.root),
                env=env,
            )

            available = result.returncode == 0
            error = result.stderr[:200].strip() if not available else None

            session.providers[name] = ProviderStatus(
                available=available,
                last_check=now_iso_utc(),
                error=error,
            )
            if name == "claude":
                session.claude_logged_in = bool(available)
            if name == "gemini":
                session.gemini_cli_logged_in = bool(available)

        except subprocess.TimeoutExpired:
            session.providers[name] = ProviderStatus(
                available=False,
                last_check=now_iso_utc(),
                error="Timeout",
            )
            if name == "claude":
                session.claude_logged_in = False
            if name == "gemini":
                session.gemini_cli_logged_in = False
        except Exception as e:
            session.providers[name] = ProviderStatus(
                available=False,
                last_check=now_iso_utc(),
                error=str(e),
            )
            if name == "claude":
                session.claude_logged_in = False
            if name == "gemini":
                session.gemini_cli_logged_in = False

        # Only emit event if status changed
        old_status = session.providers.get(name)
        self.save()

        # Emit bus event only if status changed
        current_status = session.providers[name]
        if not old_status or old_status.available != current_status.available or old_status.error != current_status.error:
            self._emit_bus("dashboard.vps.provider_status", {
                "provider": name,
                "available": current_status.available,
                "error": current_status.error,
            })

        return session.providers[name]

    def refresh_all_providers(self) -> dict[str, ProviderStatus]:
        """Refresh all providers."""
        session = self.load()

        for name in session.providers.keys():
            self.refresh_provider(name)

        return session.providers

    def determine_active_fallback(self) -> str | None:
        """Determine which provider in the fallback chain is active."""
        session = self.load()
        prev = session.active_fallback
        now = time.time()
        allow_mock = (os.environ.get("PLURIBUS_ALLOW_MOCK") or "").strip().lower() in {"1", "true", "yes", "on"}

        for provider in session.fallback_order:
            if provider == "mock" and not allow_mock:
                continue
            # Map fallback names to provider names
            # Map fallback names to provider names (vertex-gemini-curl per iTerm2 handoff)
            provider_map = {
                "codex-cli": "codex",
                "gemini": "gemini",
                "gemini-cli": "gemini",
                "vertex-gemini": "vertex",
                "vertex-gemini-curl": "vertex-curl",
                "claude-api": "claude",
                "claude-cli": "claude",
                "mock": "mock",
            }

            provider_name = provider_map.get(provider, provider)
            status = session.providers.get(provider_name)
            cooldown_until = session.provider_cooldowns.get(provider_name)
            if cooldown_until and now < cooldown_until:
                continue

            if status and status.available:
                session.active_fallback = provider
                self.save()
                if provider != prev:
                    self._emit_bus("dashboard.vps.fallback_activated", {"fallback": provider})
                return provider

        # If nothing is available, don't silently fall back to mock (internal-only).
        session.active_fallback = None
        self.save()
        if prev is not None:
            self._emit_bus("dashboard.vps.fallback_activated", {"fallback": None})
        return None

    def apply_provider_incident(self, provider: str, *, available: bool, error: str | None, cooldown_s: float | None = None) -> None:
        """
        Apply a runtime incident to control-plane state (e.g. overload, quota, auth).
        This is intended to be driven by append-only bus events.
        """
        session = self.load()
        name = (provider or "").strip()
        if not name:
            return

        provider_map = {
            "codex-cli": "codex",
            "codex": "codex",
            "gemini-cli": "gemini",
            "gemini": "gemini",
            "vertex-gemini": "vertex",
            "vertex": "vertex",
            "vertex-gemini-curl": "vertex-curl",
            "vertex-curl": "vertex-curl",
            "claude-api": "claude",
            "claude-cli": "claude",
            "claude": "claude",
            "mock": "mock",
        }
        provider_name = provider_map.get(name, name)
        st = session.providers.get(provider_name)
        if not st:
            return

        st.available = bool(available)
        st.last_check = now_iso_utc()
        st.error = error
        session.providers[provider_name] = st

        if cooldown_s and cooldown_s > 0:
            session.provider_cooldowns[provider_name] = time.time() + float(cooldown_s)
        elif cooldown_s == 0:
            session.provider_cooldowns.pop(provider_name, None)

        self.save()
        self.determine_active_fallback()

        self._emit_bus(
            "dashboard.vps.provider_status",
            {
                "provider": provider_name,
                "available": bool(st.available),
                "error": st.error,
                "model": st.model,
                "browser": st.browser,
                "cooldown_until": session.provider_cooldowns.get(provider_name),
            },
        )

    def _emit_bus(self, topic: str, data: dict) -> None:
        """Emit an event to the agent bus."""
        bus_tool = self.tools_dir / "agent_bus.py"
        if not bus_tool.exists():
            return

        bus_dir = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
        if not bus_dir:
            return

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(bus_tool),
                    "--bus-dir",
                    bus_dir,
                    "pub",
                    "--topic", topic,
                    "--kind", "metric",
                    "--level", "info",
                    "--actor",
                    os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "vps-session",
                    "--data", json.dumps(data),
                ],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    mgr = VPSSessionManager(root)
    session = mgr.load()

    print("VPS Session Status")
    print("=" * 50)
    print(f"Flow Mode: {session.flow_mode} ({'Monitor/Approve' if session.flow_mode == 'm' else 'Automatic'})")
    print()

    print("Providers:")
    for name, status in session.providers.items():
        indicator = "[+]" if status.available else "[-]"
        error_str = f" ({status.error})" if status.error else ""
        print(f"  {indicator} {name}: {'Available' if status.available else 'Unavailable'}{error_str}")

    print()
    print(f"Fallback Order: {' → '.join(session.fallback_order)}")
    print(f"Active Fallback: {session.active_fallback or 'none'}")

    print()
    print("Authentication:")
    print(f"  Gemini CLI: {'Logged in' if session.gemini_cli_logged_in else 'Not logged in'}")
    print(f"  Claude Code: {'Logged in' if session.claude_logged_in else 'Not logged in'}")
    if session.gcp_project:
        print(f"  GCP Project: {session.gcp_project}")
        print(f"  GCP Location: {session.gcp_location or 'default'}")

    if args.json:
        print()
        print(json.dumps({
            "flow_mode": session.flow_mode,
            "providers": {k: asdict(v) for k, v in session.providers.items()},
            "fallback_order": session.fallback_order,
            "active_fallback": session.active_fallback,
        }, indent=2))

    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """
    Bus-controlled VPS session daemon.

    Watches the bus for:
      - dashboard.vps.refresh_providers
      - dashboard.vps.set_flow_mode {mode:m|A}

    Also refreshes providers periodically (interval_s).
    """
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    mgr = VPSSessionManager(root)
    bus_dir = (args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
    if not bus_dir:
        print("missing PLURIBUS_BUS_DIR (set env or pass --bus-dir)", file=sys.stderr)
        return 2

    events_path = Path(bus_dir) / "events.ndjson"
    ensure_dir(events_path.parent)
    if not events_path.exists():
        events_path.write_text("", encoding="utf-8")

    actor = os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "vps-session"
    mgr._emit_bus("dashboard.vps.daemon", {"status": "starting", "iso": now_iso_utc(), "actor": actor})

    def do_refresh() -> None:
        mgr.refresh_all_providers()
        active = mgr.determine_active_fallback()
        session = mgr.load()
        # Emit per-provider status (web dashboard reducer listens to this topic).
        for name, st in session.providers.items():
            mgr._emit_bus(
                "dashboard.vps.provider_status",
                {
                    "provider": name,
                    "available": bool(st.available),
                    "error": st.error,
                    "model": st.model,
                    "cooldown_until": session.provider_cooldowns.get(name),
                },
            )
        mgr._emit_bus("dashboard.vps.refresh_complete", {"active_fallback": active, "iso": now_iso_utc()})

    # Backfill a short window so a dashboard command emitted right before startup is not missed.
    backfill_s = float(getattr(args, "backfill_s", 5.0) or 5.0)
    start_ts = time.time() - max(0.0, backfill_s)
    tail = _read_tail_bytes(events_path, max_bytes=512 * 1024)
    for raw in tail.splitlines()[-2000:]:
        try:
            obj = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            continue
        try:
            ts = float(obj.get("ts") or 0.0)
        except Exception:
            ts = 0.0
        if ts < start_ts:
            continue
        topic = str(obj.get("topic") or "")
        data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
        if topic == "dashboard.vps.refresh_providers":
            do_refresh()
        elif topic == "dashboard.vps.set_flow_mode":
            mode = str((data or {}).get("mode") or "")
            if mode in {"m", "A"}:
                mgr.set_flow_mode(mode)
        elif topic == "providers.incident":
            provider = str((data or {}).get("provider") or "")
            available = bool((data or {}).get("available", False))
            error = (data or {}).get("error")
            cooldown_s = (data or {}).get("cooldown_s")
            try:
                cooldown_s_f = float(cooldown_s) if cooldown_s is not None else None
            except Exception:
                cooldown_s_f = None
            mgr.apply_provider_incident(
                provider,
                available=available,
                error=str(error) if error is not None else None,
                cooldown_s=cooldown_s_f,
            )

    # Follow new bus lines from the end.
    interval_s = max(1.0, float(args.interval_s))
    next_refresh = time.time() + interval_s

    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            if time.time() >= next_refresh:
                do_refresh()
                next_refresh = time.time() + interval_s

            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            topic = str(obj.get("topic") or "")
            data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
            if topic == "dashboard.vps.refresh_providers":
                do_refresh()
            elif topic == "dashboard.vps.set_flow_mode":
                mode = str((data or {}).get("mode") or "")
                if mode in {"m", "A"}:
                    mgr.set_flow_mode(mode)
            elif topic == "providers.incident":
                provider = str((data or {}).get("provider") or "")
                available = bool((data or {}).get("available", False))
                error = (data or {}).get("error")
                cooldown_s = (data or {}).get("cooldown_s")
                try:
                    cooldown_s_f = float(cooldown_s) if cooldown_s is not None else None
                except Exception:
                    cooldown_s_f = None
                mgr.apply_provider_incident(
                    provider,
                    available=available,
                    error=str(error) if error is not None else None,
                    cooldown_s=cooldown_s_f,
                )


def cmd_refresh(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    mgr = VPSSessionManager(root)

    print("Refreshing provider status...")

    if args.provider:
        status = mgr.refresh_provider(args.provider)
        indicator = "[+]" if status.available else "[-]"
        print(f"  {indicator} {args.provider}: {'Available' if status.available else 'Unavailable'}")
    else:
        statuses = mgr.refresh_all_providers()
        for name, status in statuses.items():
            indicator = "[+]" if status.available else "[-]"
            print(f"  {indicator} {name}: {'Available' if status.available else 'Unavailable'}")

    # Determine active fallback
    active = mgr.determine_active_fallback()
    print(f"\nActive fallback: {active}")

    return 0


def cmd_mode(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    mgr = VPSSessionManager(root)

    try:
        mgr.set_flow_mode(args.mode)
        mode_name = "Monitor/Approve" if args.mode == "m" else "Automatic"
        print(f"Flow mode set to: {args.mode} ({mode_name})")
        return 0
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="VPS Session Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Common parent parser
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--root", help="Pluribus root directory")
    parent_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # status
    subparsers.add_parser("status", parents=[parent_parser], help="Show session status")

    # refresh
    refresh_parser = subparsers.add_parser("refresh", parents=[parent_parser], help="Refresh provider status")
    refresh_parser.add_argument("--provider", help="Specific provider to refresh")

    # mode
    mode_parser = subparsers.add_parser("mode", parents=[parent_parser], help="Set flow mode")
    mode_parser.add_argument("mode", choices=["m", "A"], help="Flow mode (m=monitor, A=automatic)")

    # daemon
    daemon_parser = subparsers.add_parser("daemon", parents=[parent_parser], help="Run bus-controlled daemon (refresh + mode).")
    daemon_parser.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    daemon_parser.add_argument("--interval-s", default="30", help="Periodic refresh interval seconds (default: 30).")
    daemon_parser.add_argument("--backfill-s", default="5", help="Backfill seconds for early dashboard commands (default: 5).")

    args = parser.parse_args()

    if not args.command:
        args = parser.parse_args(["status"])

    commands = {
        "status": cmd_status,
        "refresh": cmd_refresh,
        "mode": cmd_mode,
        "daemon": cmd_daemon,
    }

    return commands.get(args.command, cmd_status)(args)


if __name__ == "__main__":
    sys.exit(main())
