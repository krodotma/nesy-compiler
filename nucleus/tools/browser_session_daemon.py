#!/usr/bin/env python3
"""Browser Session Daemon - Persistent browser tabs for web OAuth providers.

Manages a single Chromium browser instance with persistent tabs for:
- Gemini Web (Google AI Studio)
- Claude Web (claude.ai)
- ChatGPT Web (chat.openai.com)

Features:
- Persistent browser context (cookies/sessions survive restarts)
- Tab pool management with health monitoring
- Auto-refresh before session expiry (~24hr)
- Chat transcript capture → bus events for KG/vector ingest
- VNC mode for manual OAuth login via visible browser

VNC Mode:
- Set PLURIBUS_BROWSER_VNC=1 to enable VNC-aware mode
- When VNC is running (DISPLAY is set), browser launches in visible mode
- User can connect via VNC and manually complete OAuth login
- Sessions persist after login for automated use

Usage:
    python3 browser_session_daemon.py start      # Start daemon
    python3 browser_session_daemon.py status     # Check tab health
    python3 browser_session_daemon.py stop       # Stop daemon
    python3 browser_session_daemon.py infer <provider> <prompt>  # Test inference
    python3 browser_session_daemon.py vnc-mode start   # Enable VNC mode
    python3 browser_session_daemon.py vnc-mode stop    # Disable VNC mode
    python3 browser_session_daemon.py vnc-mode status  # Check VNC mode status
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.parse import urlparse
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Browser Guardian - single-process enforcement
try:
    from browser_guardian import BrowserGuardian
    _guardian = BrowserGuardian()
except ImportError:
    _guardian = None
from typing import Any, Optional

# Stealth mode for better bot evasion
try:
    from playwright_stealth import Stealth
    STEALTH = Stealth(
        navigator_webdriver=True,  # Hide webdriver flag
        chrome_app=True,
        chrome_csi=True,
        chrome_load_times=True,
        navigator_languages=True,
        navigator_platform=True,
        navigator_plugins=True,
        navigator_permissions=True,
        webgl_vendor=True,
        media_codecs=True,
    )
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH = None
    STEALTH_AVAILABLE = False

sys.dont_write_bytecode = True

# Best-effort import for structured bus emission.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from nucleus.tools import agent_bus  # type: ignore
except Exception:  # pragma: no cover
    agent_bus = None  # type: ignore

# Bus integration
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def _kind_for_topic(topic: str) -> str:
    t = (topic or "").strip()
    if t.endswith(".request"):
        return "request"
    if t.endswith(".response"):
        return "response"
    if t.endswith(".completed") or t.endswith(".list") or t.endswith(".artifact"):
        return "artifact"
    return "metric"


# Elite patch v1.0: Provider-to-actor mapping for accurate bus tracing
PROVIDER_TO_ACTOR = {
    "claude-web": "claude",
    "gemini-web": "gemini", 
    "chatgpt-web": "chatgpt",
    "grok-web": "grok",
    "ollama-local": "ollama",
    "vllm-local": "vllm",
    "codex": "codex",
}

def provider_to_actor(provider: str, default: str = "browser_daemon") -> str:
    """Convert provider name to canonical actor for bus emission."""
    if not provider:
        return default
    return PROVIDER_TO_ACTOR.get(provider, provider)



def append_bus_event(
    bus_dir: Path,
    topic: str,
    data: dict,
    actor: str = "browser_daemon",
    *,
    kind: str | None = None,
    level: str = "info",
) -> None:
    """Append event to Pluribus bus (best-effort structured emission)."""
    kind = kind or _kind_for_topic(topic)
    if agent_bus is not None:
        try:
            paths = agent_bus.resolve_bus_paths(str(bus_dir))
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind=kind,
                level=level,
                actor=actor,
                data=data,
                trace_id=None,
                run_id=None,
                durable=False,
            )
            return
        except Exception:
            pass

    # Fallback: minimal NDJSON append compatible with agent_bus schema.
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _proc_cmdline(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _proc_state_code(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{int(pid)}/status").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    for line in raw.splitlines():
        if not line.startswith("State:"):
            continue
        state = line.split(":", 1)[1].strip()
        if not state:
            return None
        return state.split(None, 1)[0].strip()
    return None


def _is_zombie_pid(pid: int) -> bool:
    return _proc_state_code(pid) == "Z"


def _is_browser_session_daemon_pid(pid: int) -> bool:
    pid = int(pid or 0)
    if pid <= 0:
        return False
    if not Path(f"/proc/{pid}").exists():
        return False
    if _is_zombie_pid(pid):
        return False
    cmdline = _proc_cmdline(pid) or ""
    return "browser_session_daemon.py" in cmdline

def _find_browser_pid(user_data_dir: Path) -> int:
    target = f"--user-data-dir={user_data_dir}"
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            cmdline = _proc_cmdline(pid)
            if not cmdline or target not in cmdline:
                continue
            if ("chrome" not in cmdline) and ("chromium" not in cmdline):
                continue
            if "--type=" in cmdline and "--type=browser" not in cmdline:
                continue
            return pid
    except Exception:
        return 0
    return 0

def _chatgpt_noauth_gate(html: str) -> bool:
    """
    Detect ChatGPT's logged-out gate modal ("Thanks for trying ChatGPT").

    When present, Playwright chat interactions can fail with misleading selector/empty-response errors.
    We classify this as needs_login so the control plane can surface an actionable status.
    """
    t = (html or "")
    if 'data-testid="modal-no-auth-rate-limit"' in t:
        return True
    if ("Thanks for trying ChatGPT" in t) and ("Stay logged out" in t):
        return True
    return False


# ============================================================================
# URL Classification Helpers
# ============================================================================

def _url_host(url: str) -> str:
    """Best-effort host extraction (lowercased)."""
    u = (url or "").strip()
    if not u:
        return ""
    try:
        return (urlparse(u).hostname or "").strip().lower()
    except Exception:
        return ""


def _allowed_hosts_for_config(config: dict) -> set[str]:
    """Return the set of allowed hosts for a provider config."""
    allowed = config.get("allowed_hosts")
    if isinstance(allowed, (list, tuple, set)):
        hosts = {str(h).strip().lower() for h in allowed if str(h).strip()}
        if hosts:
            return hosts
    host = _url_host(str(config.get("url") or ""))
    return {host} if host else set()


def _looks_like_claude_service_disruption(html: str) -> bool:
    """Detect Anthropic temporary outage / maintenance pages."""
    t = (html or "").lower()
    # Observed outage copy (keep checks broad but low false-positive).
    if "claude will return soon" in t:
        return True
    if "temporary service disruption" in t:
        return True
    if "reaching out to support" in t and "error code" in t:
        return True
    return False


def _looks_like_login_flow(url: str, *, login_url: str | None = None) -> bool:
    """Heuristically classify whether a URL is part of an auth/login flow."""
    u = (url or "").strip().lower()
    if not u:
        return False
    lu = (login_url or "").strip().lower()
    if lu and lu in u:
        return True
    markers = (
        "login",
        "signin",
        "sign-in",
        "auth",
        "oauth",
        "accounts.google.com",
        "auth0",
        "session",
        "rejected",
    )
    return any(m in u for m in markers)


# ============================================================================
# VNC Mode Detection and Configuration
# ============================================================================

def is_vnc_mode_enabled() -> bool:
    """Check if VNC mode is enabled via environment variable."""
    val = os.environ.get("PLURIBUS_BROWSER_VNC", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def detect_display() -> Optional[str]:
    """Detect if DISPLAY environment variable is set (indicates X11/VNC available).

    Returns the DISPLAY value if set, None otherwise.
    """
    display = os.environ.get("DISPLAY", "").strip()
    return display if display else None


def _parse_xrandr_screen(raw: str) -> Optional[tuple[int, int]]:
    match = re.search(r"current\s+(\d+)\s+x\s+(\d+)", raw)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_xwininfo_root(raw: str) -> Optional[tuple[int, int]]:
    width = None
    height = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Width:"):
            try:
                width = int(line.split(":", 1)[1].strip())
            except Exception:
                width = None
        elif line.startswith("Height:"):
            try:
                height = int(line.split(":", 1)[1].strip())
            except Exception:
                height = None
    if width and height:
        return width, height
    return None


def detect_screen_size(display: Optional[str]) -> Optional[tuple[int, int]]:
    if not display:
        return None
    env = dict(os.environ)
    env["DISPLAY"] = display
    for cmd, parser in ((["xrandr", "--current"], _parse_xrandr_screen), (["xwininfo", "-root"], _parse_xwininfo_root)):
        try:
            raw = subprocess.check_output(cmd, env=env, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            continue
        parsed = parser(raw)
        if parsed:
            return parsed
    return None


def _x11_list_windows(env: dict[str, str]) -> list[int]:
    try:
        raw = subprocess.check_output(["xprop", "-root", "_NET_CLIENT_LIST"], env=env, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return []
    return [int(token, 16) for token in re.findall(r"0x[0-9a-fA-F]+", raw)]


def _x11_get_wm_class(env: dict[str, str], win_id: int) -> str:
    try:
        raw = subprocess.check_output(["xprop", "-id", hex(win_id), "WM_CLASS"], env=env, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return ""
    parts = re.findall(r"\"([^\"]+)\"", raw)
    return (parts[-1] if parts else "").strip().lower()


def _x11_find_browser_windows(env: dict[str, str]) -> dict[str, int]:
    windows: dict[str, int] = {}
    for win_id in reversed(_x11_list_windows(env)):
        wm_class = _x11_get_wm_class(env, win_id)
        if not wm_class:
            continue
        if "firefox" in wm_class and "firefox" not in windows:
            windows["firefox"] = win_id
        elif ("chrome" in wm_class or "chromium" in wm_class) and "chromium" not in windows:
            windows["chromium"] = win_id
        if "firefox" in windows and "chromium" in windows:
            break
    return windows


def _x11_move_resize_window(win_id: int, *, x: int, y: int, width: int, height: int) -> bool:
    try:
        x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
    except Exception:
        return False
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XMoveResizeWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(None)
    if not display:
        return False
    try:
        x11.XMoveResizeWindow(display, ctypes.c_ulong(win_id), x, y, width, height)
        x11.XFlush(display)
        return True
    finally:
        x11.XCloseDisplay(display)


def _compute_split_layout(
    screen_width: int,
    screen_height: int,
    mode: str,
    ratio: float,
    primary: str,
) -> dict[str, tuple[int, int, int, int]]:
    ratio = max(0.1, min(0.9, ratio))
    layout: dict[str, tuple[int, int, int, int]] = {}
    if mode == "horizontal":
        top_h = int(screen_height * ratio)
        bottom_h = max(1, screen_height - top_h)
        primary_geom = (0, 0, screen_width, top_h)
        secondary_geom = (0, top_h, screen_width, bottom_h)
    else:
        left_w = int(screen_width * ratio)
        right_w = max(1, screen_width - left_w)
        primary_geom = (0, 0, left_w, screen_height)
        secondary_geom = (left_w, 0, right_w, screen_height)

    if primary == "chromium":
        layout["chromium"] = primary_geom
        layout["firefox"] = secondary_geom
    else:
        layout["firefox"] = primary_geom
        layout["chromium"] = secondary_geom
    return layout


def _apply_split_layout(display: str, layout: dict[str, tuple[int, int, int, int]]) -> bool:
    env = dict(os.environ)
    env["DISPLAY"] = display
    windows = _x11_find_browser_windows(env)
    if not windows:
        return False
    ok = False
    for browser, geom in layout.items():
        win_id = windows.get(browser)
        if not win_id:
            continue
        x, y, width, height = geom
        if _x11_move_resize_window(win_id, x=x, y=y, width=width, height=height):
            ok = True
    return ok


def is_vnc_server_running() -> bool:
    """Check if a VNC server is running by looking for common VNC processes."""
    vnc_processes = ["Xvnc", "x11vnc", "tigervncserver", "vncserver", "Xtigervnc"]
    for proc_name in vnc_processes:
        try:
            result = subprocess.run(
                ["pgrep", "-x", proc_name],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    # Also check if DISPLAY points to a VNC display (typically :1, :2, etc.)
    display = detect_display()
    if display and display.startswith(":") and display[1:].split(".")[0].isdigit():
        disp_num = int(display[1:].split(".")[0])
        if disp_num >= 1:  # :0 is usually physical display, :1+ are virtual
            return True

    return False


def get_vnc_connection_info() -> dict:
    """Get VNC connection information for user guidance."""
    display = detect_display()
    hostname = os.environ.get("HOSTNAME", "localhost")

    # Try to determine VNC port from display number
    vnc_port = 5900
    if display and display.startswith(":"):
        try:
            disp_num = int(display[1:].split(".")[0])
            vnc_port = 5900 + disp_num
        except Exception:
            pass

    return {
        "display": display,
        "hostname": hostname,
        "vnc_port": vnc_port,
        "connection_string": f"{hostname}:{vnc_port}",
        "instructions": f"Connect via VNC to {hostname}:{vnc_port} to interact with the browser"
    }


@dataclass
class VNCModeState:
    """VNC mode state tracking."""
    enabled: bool = False
    display: Optional[str] = None
    vnc_detected: bool = False
    started_at: Optional[str] = None
    login_providers_pending: list = field(default_factory=list)
    login_providers_completed: list = field(default_factory=list)


@dataclass
class TabSession:
    """State of a browser tab session."""
    provider_id: str
    tab_id: str
    url: str
    status: str = "initializing"  # initializing, ready, busy, error, closed, needs_login, needs_onboarding, needs_code, needs_project, blocked_bot
    last_health_check: str = ""
    last_activity: str = ""
    session_start: str = ""
    current_url: str = ""
    title: str = ""
    error: Optional[str] = None
    chat_count: int = 0


@dataclass
class BrowserDaemonState:
    """Daemon state persisted to disk."""
    running: bool = False
    pid: int = 0
    browser_pids: dict[str, int] = field(default_factory=dict)
    started_at: str = ""
    tabs: dict[str, TabSession] = None
    vnc_mode: VNCModeState = None

    def __post_init__(self):
        if self.tabs is None:
            self.tabs = {}
        if self.vnc_mode is None:
            self.vnc_mode = VNCModeState()
        if self.browser_pids is None:
            self.browser_pids = {}


# Provider configurations
WEB_PROVIDERS = {
    "gemini-web": {
        "name": "Gemini Web",
        "description": "Google AI Studio - Edge Inference",
        "url": "https://aistudio.google.com/",
        "login_url": "https://accounts.google.com/",
        "allowed_hosts": ["aistudio.google.com", "ai.google.dev"],
        "chat_selector": "textarea[aria-label*='prompt' i], textarea[placeholder*='prompt' i], textarea",
        "submit_selector": "button[aria-label*='send'], button[type='submit']",
        "response_selector": ".response-content, .model-response, [data-message-author-role='model']",
        "max_session_hours": 24,
        "browser_type": "firefox",
        "stealth": False,
    },
    "claude-web": {
        "name": "Claude Web",
        "description": "Anthropic Claude - Edge Inference",
        "url": "https://claude.ai/new",
        "login_url": "https://claude.ai/login",
        "allowed_hosts": ["claude.ai"],
        "chat_selector": "textarea[placeholder*='message'], div[contenteditable='true']",
        "submit_selector": "button[aria-label*='send'], button[type='submit']",
        "response_selector": ".assistant-message, [data-role='assistant']",
        "max_session_hours": 24,
        "browser_type": "chromium",
        "stealth": True,
    },
    "chatgpt-web": {
        "name": "ChatGPT Web",
        "description": "OpenAI ChatGPT - Edge Inference",
        "url": "https://chat.openai.com/",
        "login_url": "https://auth0.openai.com/",
        "allowed_hosts": ["chat.openai.com", "chatgpt.com"],
        "chat_selector": "textarea[data-id='root'], #prompt-textarea",
        "submit_selector": "button[data-testid='send-button'], button[type='submit']",
        "response_selector": "[data-message-author-role='assistant']",
        "max_session_hours": 24,
        "browser_type": "chromium",
        "stealth": True,
    },
    # Grok Web - xAI's chat interface (SuperGrok subscription recommended)
    # Auth: X.com OAuth (Twitter/X account required)
    # Limits: 100 msgs/2hr (SuperGrok), 30 Think/2hr, 30 DeepSearch/2hr
    # Models: Grok 4 (Expert mode), Grok 3 (Fast mode) - no model selector in UI
    # Note: Selectors are provisional - inspect grok.com before enabling
    "grok-web": {
        "name": "Grok Web",
        "description": "xAI Grok - Edge Inference (SuperGrok)",
        "url": "https://grok.com/",
        "login_url": "https://x.com/i/flow/login",
        "allowed_hosts": ["grok.com", "x.com", "twitter.com"],
        # Provisional selectors - need inspection of grok.com DOM
        "chat_selector": "textarea[placeholder*='message' i], textarea[aria-label*='message' i], div[contenteditable='true']",
        "submit_selector": "button[aria-label*='send' i], button[type='submit'], button svg[class*='send' i]",
        "response_selector": "[data-testid='conversation-turn'], .grok-response, [role='article']",
        "max_session_hours": 24,
        "browser_type": "chromium",
        "stealth": True,  # X.com has aggressive bot detection
        # Grok-specific: X OAuth uses different flow than Google/OpenAI
        "oauth_provider": "x.com",
        "requires_2fa": True,  # X.com often requires 2FA
    },
}

# Dashboard configuration (for CUA auto-reload)
DASHBOARD_CONFIG = {
    # Public-facing default (override with PLURIBUS_DASHBOARD_URL for local dev).
    "url": os.environ.get("PLURIBUS_DASHBOARD_URL", "https://kroma.live"),
    "error_threshold": 5,  # errors in window before reload
    "error_window_s": 60,  # sliding window for error counting
    "reload_cooldown_s": 30,  # min seconds between reloads
}

# Auto-login knobs (env-driven; never persist secrets)
AUTOLOGIN_ENABLED = (os.environ.get("PLURIBUS_BROWSER_AUTOLOGIN") or "1").strip().lower() in {"1", "true", "yes", "on"}
AUTOLOGIN_COOLDOWN_S = int(os.environ.get("PLURIBUS_BROWSER_AUTOLOGIN_COOLDOWN") or "300")
AUTOLOGIN_DISABLED_PROVIDERS = {
    p.strip()
    for p in (os.environ.get("PLURIBUS_BROWSER_AUTOLOGIN_DISABLED") or "").split(",")
    if p.strip()
}

# Solver knobs (off by default for safety)
SOLVER_CMD = os.environ.get("PLURIBUS_SOLVER_CMD", "").strip()  # optional external command template
SOLVER_ENABLED = bool(SOLVER_CMD)
SOLVER_TIMEOUT_S = int(os.environ.get("PLURIBUS_SOLVER_TIMEOUT") or "45")


def _env_creds_for_provider(provider_id: str) -> tuple[str | None, str | None]:
    """Fetch provider credentials from environment (in-memory only).

    Supports multiple naming conventions:
    - PROVIDER_WEB_EMAIL / PROVIDER_WEB_PASS (e.g., GEMINI_WEB_EMAIL)
    - PLURIBUS_* variants
    - Legacy variants (OPENAI_USER, etc.)
    """
    if provider_id == "chatgpt-web":
        # Check all possible env var names
        u = (os.environ.get("CHATGPT_WEB_EMAIL") or
             os.environ.get("PLURIBUS_OPENAI_USER") or
             os.environ.get("OPENAI_USER") or
             os.environ.get("PLURIBUS_BROWSER_USER"))
        p = (os.environ.get("CHATGPT_WEB_PASS") or
             os.environ.get("PLURIBUS_OPENAI_PASS") or
             os.environ.get("OPENAI_PASS") or
             os.environ.get("PLURIBUS_BROWSER_PASS"))
        return u, p
    if provider_id == "claude-web":
        u = (os.environ.get("CLAUDE_WEB_EMAIL") or
             os.environ.get("PLURIBUS_CLAUDE_USER") or
             os.environ.get("PLURIBUS_GOOGLE_USER") or  # Claude federates to Google
             os.environ.get("PLURIBUS_BROWSER_USER"))
        p = (os.environ.get("CLAUDE_WEB_PASS") or
             os.environ.get("PLURIBUS_CLAUDE_PASS") or
             os.environ.get("PLURIBUS_GOOGLE_PASS") or
             os.environ.get("PLURIBUS_BROWSER_PASS"))
        return u, p
    if provider_id == "gemini-web":
        u = (os.environ.get("GEMINI_WEB_EMAIL") or
             os.environ.get("PLURIBUS_GOOGLE_USER") or
             os.environ.get("GOOGLE_USER") or
             os.environ.get("PLURIBUS_BROWSER_USER"))
        p = (os.environ.get("GEMINI_WEB_PASS") or
             os.environ.get("PLURIBUS_GOOGLE_PASS") or
             os.environ.get("GOOGLE_PASS") or
             os.environ.get("PLURIBUS_BROWSER_PASS"))
        return u, p
    if provider_id == "grok-web":
        # X.com OAuth - uses username (not email) or phone
        u = (os.environ.get("GROK_WEB_USER") or
             os.environ.get("PLURIBUS_X_USER") or
             os.environ.get("X_USER") or
             os.environ.get("TWITTER_USER") or
             os.environ.get("PLURIBUS_BROWSER_USER"))
        p = (os.environ.get("GROK_WEB_PASS") or
             os.environ.get("PLURIBUS_X_PASS") or
             os.environ.get("X_PASS") or
             os.environ.get("TWITTER_PASS") or
             os.environ.get("PLURIBUS_BROWSER_PASS"))
        return u, p
    return None, None


def _run_solver(
    cmd_template: str,
    html_path: Path,
    screenshot_path: Path,
    provider_id: str,
    *,
    reason: str | None = None,
    req_id: str | None = None,
) -> str | None:
    """
    Best-effort external solver hook. Returns solver output (stdout) or None.
    """
    if not cmd_template:
        return None
    tpl = cmd_template.replace("{provider}", provider_id)
    tpl = tpl.replace("{html}", str(html_path))
    tpl = tpl.replace("{screenshot}", str(screenshot_path))
    tpl = tpl.replace("{reason}", str(reason or ""))
    tpl = tpl.replace("{req_id}", str(req_id or ""))
    try:
        res = subprocess.run(
            tpl,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=SOLVER_TIMEOUT_S,
        )
        out = (res.stdout or "").strip()
        return out or None
    except Exception:
        return None


def _generate_totp(secret: str) -> str | None:
    """
    Generate a 6-digit TOTP code.

    Args:
        secret: Base32-encoded TOTP secret (from Google Authenticator setup)

    Returns:
        6-digit TOTP code or None if generation fails
    """
    if not secret:
        return None
    secret = secret.strip()
    if not secret:
        return None

    # Prefer oathtool if available (matches common operator setups).
    try:
        result = subprocess.run(
            ["oathtool", "--totp", "--base32", secret],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    # Fallback: pure-Python TOTP (RFC 6238 / HOTP with 30s timestep, 6 digits).
    try:
        import base64
        import hashlib
        import hmac
        import struct

        normalized = secret.replace(" ", "")
        # Pad to a multiple of 8 for base32 decode.
        missing = (-len(normalized)) % 8
        if missing:
            normalized += "=" * missing
        key = base64.b32decode(normalized, casefold=True)
        counter = int(time.time() // 30)
        msg = struct.pack(">Q", counter)
        digest = hmac.new(key, msg, hashlib.sha1).digest()
        offset = digest[-1] & 0x0F
        code_int = (struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF) % 1_000_000
        return f"{code_int:06d}"
    except Exception:
        return None
    return None


def _get_totp_secret_for_provider(provider_id: str) -> str | None:
    """Get TOTP secret from environment for a provider."""
    if provider_id in ("gemini-web", "claude-web"):
        # Both use Google SSO
        return (os.environ.get("PLURIBUS_GOOGLE_TOTP_SECRET") or
                os.environ.get("GOOGLE_TOTP_SECRET") or
                os.environ.get("PLURIBUS_TOTP_SECRET"))
    if provider_id == "chatgpt-web":
        return (os.environ.get("PLURIBUS_OPENAI_TOTP_SECRET") or
                os.environ.get("OPENAI_TOTP_SECRET") or
                os.environ.get("PLURIBUS_TOTP_SECRET"))
    if provider_id == "grok-web":
        # X.com / Twitter OAuth TOTP for SuperGrok access
        return (os.environ.get("GROK_WEB_TOTP_SECRET") or
                os.environ.get("PLURIBUS_X_TOTP_SECRET") or
                os.environ.get("X_TOTP_SECRET") or
                os.environ.get("TWITTER_TOTP_SECRET") or
                os.environ.get("PLURIBUS_TOTP_SECRET"))
    return None


def _google_insecure_browser_gate(text: str) -> bool:
    t = (text or "").lower()
    return "this browser or app may not be secure" in t


async def _google_login_flow(
    page: Any,
    user: str,
    pw: str,
    *,
    start_url: str | None = None,
    timeout_s: int = 45,
) -> dict:
    """
    Best-effort Google login flow (email -> next -> password -> next).
    Returns dict: success, needs_code, blocked_insecure, message.
    """
    if start_url:
        try:
            await page.goto(start_url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
            await asyncio.sleep(0.75)
        except Exception:
            pass
    try:
        # Email/identifier
        email_sel = 'input[type="email"], input[name="identifier"], input[id="identifierId"]'
        el = await page.wait_for_selector(email_sel, timeout=5000)
        await el.fill(user)
        await page.keyboard.press("Enter")
        await asyncio.sleep(1.5)
    except Exception:
        pass
    try:
        # Password
        pw_sel = 'input[type="password"], input[name="Passwd"]'
        pw_el = await page.wait_for_selector(pw_sel, timeout=8000)
        await pw_el.fill(pw)
        await page.keyboard.press("Enter")
        await asyncio.sleep(2.5)
    except Exception:
        pass
    # Detect OTP/verification prompt
    needs_code = False
    blocked_insecure = False
    totp_auto_filled = False
    try:
        body_text = (await page.text_content("body")) or ""
        blocked_insecure = _google_insecure_browser_gate(body_text)
        if any(k in body_text.lower() for k in ["verification code", "enter code", "2-step", "2 step", "two-factor", "otp"]):
            needs_code = True
            # Try auto-fill TOTP if secret is available
            totp_secret = _get_totp_secret_for_provider("gemini-web")  # Use gemini-web for Google SSO
            if totp_secret:
                totp_code = _generate_totp(totp_secret)
                if totp_code:
                    try:
                        # Common Google OTP input selectors
                        otp_selectors = [
                            'input[name="totpPin"]',
                            'input[type="tel"]',
                            'input[aria-label*="code"]',
                            'input[aria-label*="verification"]',
                            'input[autocomplete="one-time-code"]',
                        ]
                        for sel in otp_selectors:
                            try:
                                otp_el = await page.wait_for_selector(sel, timeout=3000)
                                await otp_el.fill(totp_code)
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(2.5)
                                totp_auto_filled = True
                                needs_code = False  # Successfully auto-filled
                                break
                            except Exception:
                                continue
                    except Exception:
                        pass
    except Exception:
        pass
    return {
        "success": (not needs_code) and (not blocked_insecure),
        "needs_code": needs_code,
        "blocked_insecure": blocked_insecure,
        "totp_auto_filled": totp_auto_filled,
        "message": "insecure browser gate" if blocked_insecure else ("totp auto-filled" if totp_auto_filled else ("otp required" if needs_code else "login attempted")),
    }


async def _x_login_flow(
    page: Any,
    user: str,
    pw: str,
    *,
    start_url: str | None = None,
    timeout_s: int = 60,
) -> dict:
    """
    Best-effort X.com (Twitter) login flow for grok-web access.
    Returns dict: success, needs_code, message.
    """
    login_url = start_url or "https://x.com/i/flow/login"
    try:
        await page.goto(login_url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
        await asyncio.sleep(1.5)
    except Exception:
        pass

    # Step 1: Enter username/email/phone
    try:
        user_sel = 'input[autocomplete="username"], input[name="text"], input[type="text"]'
        el = await page.wait_for_selector(user_sel, timeout=8000)
        await el.fill(user)
        await asyncio.sleep(0.3)
        # Click Next button
        next_btns = ['[role="button"]:has-text("Next")', 'button:has-text("Next")', '[data-testid="LoginForm_Login_Button"]']
        for btn_sel in next_btns:
            try:
                btn = await page.wait_for_selector(btn_sel, timeout=2000)
                await btn.click()
                await asyncio.sleep(1.5)
                break
            except Exception:
                continue
    except Exception:
        pass

    # Step 2: Handle unusual activity check (username/phone verification)
    try:
        body_text = (await page.text_content("body")) or ""
        if "unusual login activity" in body_text.lower() or "verify your identity" in body_text.lower():
            # May need phone/email verification
            verify_input = await page.wait_for_selector('input[name="text"]', timeout=3000)
            if verify_input:
                # Try to enter the username again as verification
                await verify_input.fill(user)
                await asyncio.sleep(0.3)
                for btn_sel in ['[role="button"]:has-text("Next")', 'button:has-text("Next")']:
                    try:
                        btn = await page.wait_for_selector(btn_sel, timeout=2000)
                        await btn.click()
                        await asyncio.sleep(1.5)
                        break
                    except Exception:
                        continue
    except Exception:
        pass

    # Step 3: Enter password
    try:
        pw_sel = 'input[type="password"], input[name="password"], input[autocomplete="current-password"]'
        pw_el = await page.wait_for_selector(pw_sel, timeout=8000)
        await pw_el.fill(pw)
        await asyncio.sleep(0.3)
        # Click Log in button
        login_btns = ['[data-testid="LoginForm_Login_Button"]', '[role="button"]:has-text("Log in")', 'button:has-text("Log in")']
        for btn_sel in login_btns:
            try:
                btn = await page.wait_for_selector(btn_sel, timeout=2000)
                await btn.click()
                await asyncio.sleep(3)
                break
            except Exception:
                continue
    except Exception:
        pass

    # Step 4: Handle 2FA/TOTP
    needs_code = False
    totp_auto_filled = False
    try:
        body_text = (await page.text_content("body")) or ""
        if any(k in body_text.lower() for k in ["verification code", "authentication code", "two-factor", "2fa", "authenticator app"]):
            needs_code = True
            totp_secret = _get_totp_secret_for_provider("grok-web")
            if totp_secret:
                totp_code = _generate_totp(totp_secret)
                if totp_code:
                    try:
                        otp_selectors = [
                            'input[name="text"]',
                            'input[type="text"]',
                            'input[autocomplete="one-time-code"]',
                            'input[aria-label*="code"]',
                        ]
                        for sel in otp_selectors:
                            try:
                                otp_el = await page.wait_for_selector(sel, timeout=3000)
                                await otp_el.fill(totp_code)
                                await asyncio.sleep(0.3)
                                # Click confirm/next
                                for btn_sel in ['[role="button"]:has-text("Next")', 'button:has-text("Next")', 'button:has-text("Confirm")']:
                                    try:
                                        btn = await page.wait_for_selector(btn_sel, timeout=2000)
                                        await btn.click()
                                        await asyncio.sleep(2.5)
                                        totp_auto_filled = True
                                        needs_code = False
                                        break
                                    except Exception:
                                        continue
                                if totp_auto_filled:
                                    break
                            except Exception:
                                continue
                    except Exception:
                        pass
    except Exception:
        pass

    # Check final state
    success = False
    try:
        await asyncio.sleep(1)
        current_url = page.url
        # Success if we're on x.com main page or grok.com
        if "x.com/home" in current_url or "grok.com" in current_url:
            success = True
        elif "login" not in current_url.lower() and "flow" not in current_url.lower():
            success = True
    except Exception:
        pass

    return {
        "success": success and not needs_code,
        "needs_code": needs_code,
        "totp_auto_filled": totp_auto_filled,
        "message": "totp auto-filled" if totp_auto_filled else ("otp required" if needs_code else ("login successful" if success else "login attempted")),
    }


class BrowserSessionDaemon:
    """Manages persistent browser sessions for web providers."""

    def __init__(self, root: Path, bus_dir: Path):
        self.root = root
        # Prefer the explicit bus_dir argument (tests/sandboxes may set PLURIBUS_BUS_DIR globally).
        # Still honor PLURIBUS_BUS_DIR when the caller passed the default path.
        default_bus_dir = Path("/pluribus/.pluribus/bus")
        env_bus_dir = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
        if env_bus_dir and Path(bus_dir) == default_bus_dir:
            self.bus_dir = Path(env_bus_dir)
        else:
            self.bus_dir = Path(bus_dir)
        self.events_path = self.bus_dir / "events.ndjson"
        self.cursor_path = root / ".pluribus" / "browser_daemon.cursor"
        self.state_path = root / ".pluribus" / "browser_daemon.json"
        self.user_data_dir_root = root / ".pluribus" / "browser_data"
        self.playwright = None
        self.contexts: dict[str, Any] = {}  # browser_type -> context
        self.pages: dict[str, Any] = {}  # provider_id -> Page
        self.state = BrowserDaemonState()
        self._running = False
        # CUA dashboard monitoring
        self.dashboard_page: Any = None
        self.dashboard_errors: list[float] = []  # timestamps of recent errors
        self.last_dashboard_reload: float = 0.0
        # Ephemeral user_auths (OAuth token map)
        self.user_auths: dict[str, dict] = {}  # provider -> {token, expires, refresh_url}
        # Throttle auto-login attempts per provider
        self.last_auto_login_attempt: dict[str, float] = {}
        # Track if human has been notified (prevents email spam loop)
        # Once notified, we don't auto-login again until human acknowledges or 1 hour passes
        self.human_notified_for_login: dict[str, float] = {}
        self.HUMAN_NOTIFICATION_COOLDOWN_S = 3600  # 1 hour before re-notifying

    def load_state(self) -> BrowserDaemonState:
        """Load daemon state from disk."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                tabs = {}
                for k, v in data.get("tabs", {}).items():
                    tabs[k] = TabSession(**v)

                # Load VNC mode state
                vnc_data = data.get("vnc_mode", {})
                vnc_mode = VNCModeState(
                    enabled=vnc_data.get("enabled", False),
                    display=vnc_data.get("display"),
                    vnc_detected=vnc_data.get("vnc_detected", False),
                    started_at=vnc_data.get("started_at"),
                    login_providers_pending=vnc_data.get("login_providers_pending", []),
                    login_providers_completed=vnc_data.get("login_providers_completed", []),
                )

                self.state = BrowserDaemonState(
                    running=data.get("running", False),
                    pid=data.get("pid", 0),
                    browser_pids=data.get("browser_pids", {}),
                    started_at=data.get("started_at", ""),
                    tabs=tabs,
                    vnc_mode=vnc_mode,
                )
                # Never load persisted secrets. Browser sessions are persisted in the Playwright
                # user-data-dir; any OAuth/API tokens remain in-memory only.
                self.user_auths = {}

                # Guard against stale state when the daemon crashed or was killed.
                if self.state.running and self.state.pid:
                    if not _is_browser_session_daemon_pid(int(self.state.pid)):
                        self.state.running = False
                        self.state.pid = 0
                        self.state.browser_pids = {}
                        for tab in self.state.tabs.values():
                            if tab.status != "closed":
                                tab.status = "closed"
                                tab.error = "Daemon not running (stale state)"
                        try:
                            self.save_state()
                        except Exception:
                            pass
            except Exception as e:
                print(f"Error loading state: {e}")
                self.state = BrowserDaemonState()
        return self.state

    def save_state(self) -> None:
        """Save daemon state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Persist only non-sensitive metadata about auth injection attempts.
        user_auths_meta: dict[str, dict] = {}
        for provider_id, auth in (self.user_auths or {}).items():
            if not isinstance(auth, dict):
                continue
            meta: dict[str, object] = {}
            if "method" in auth and isinstance(auth.get("method"), str):
                meta["method"] = auth.get("method")
            if "injected_at" in auth and isinstance(auth.get("injected_at"), str):
                meta["injected_at"] = auth.get("injected_at")
            if "has_gcloud" in auth:
                meta["has_gcloud"] = bool(auth.get("has_gcloud"))
            if "type" in auth and isinstance(auth.get("type"), str):
                meta["type"] = auth.get("type")
            # Only record that a token existed, never its value.
            if "token" in auth and isinstance(auth.get("token"), str):
                meta["has_token"] = True
                meta["token_len"] = len(auth.get("token") or "")
            if meta:
                user_auths_meta[str(provider_id)] = meta

        data = {
            "running": self.state.running,
            "pid": self.state.pid,
            "browser_pids": self.state.browser_pids,
            "started_at": self.state.started_at,
            "tabs": {k: asdict(v) for k, v in self.state.tabs.items()},
            "vnc_mode": asdict(self.state.vnc_mode) if self.state.vnc_mode else {},
            "user_auths_meta": user_auths_meta,
        }
        # Atomic write so readers (dashboard bus-bridge) never observe partial JSON.
        payload = json.dumps(data, indent=2)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.state_path.name}.", suffix=".tmp", dir=str(self.state_path.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.state_path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    async def start(self) -> None:
        """Start the browser daemon."""
        from playwright.async_api import async_playwright

        self.load_state()

        # Check if already running
        if self.state.running and self.state.pid:
            if _is_browser_session_daemon_pid(int(self.state.pid)):
                print(f"Daemon already running (PID {self.state.pid})")
                return
            # Stale or PID-reused state; clear and continue.
            self.state.running = False
            self.state.pid = 0
            self.state.browser_pids = {}
            for tab in self.state.tabs.values():
                if tab.status != "closed":
                    tab.status = "closed"
                    tab.error = "Daemon not running (stale state)"
            try:
                self.save_state()
            except Exception:
                pass

        print("Starting browser session daemon...")
        self._running = True

        # Start playwright
        self.playwright = await async_playwright().start()

        # Determine headless mode based on VNC availability
        vnc_mode_enabled = is_vnc_mode_enabled()
        display = detect_display()
        vnc_server_running = is_vnc_server_running()

        # Default headless behavior
        headless_env = (os.environ.get("PLURIBUS_BROWSER_HEADLESS") or "1").strip().lower()
        headless_default = headless_env in {"1", "true", "yes", "on"}

        if vnc_mode_enabled and display:
            headless = False
            print(f"  VNC mode enabled (DISPLAY={display})")
            if vnc_server_running:
                vnc_info = get_vnc_connection_info()
                print(f"  VNC connection: {vnc_info['connection_string']}")
        else:
            headless = headless_default

        # Update VNC mode state
        self.state.vnc_mode = VNCModeState(
            enabled=vnc_mode_enabled,
            display=display,
            vnc_detected=vnc_server_running,
            started_at=now_iso() if vnc_mode_enabled else None,
            login_providers_pending=[],
            login_providers_completed=[],
        )

        needed_browsers = set(p.get("browser_type", "chromium") for p in WEB_PROVIDERS.values())
        disable_chromium = (os.environ.get("PLURIBUS_BROWSER_DISABLE_CHROMIUM") or "").strip().lower() in {"1", "true", "yes", "on"}
        if disable_chromium and "chromium" in needed_browsers:
            needed_browsers.discard("chromium")
        chromium_user_agent = os.environ.get(
            "PLURIBUS_CHROMIUM_USER_AGENT",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.7499.4 Safari/537.36",
        )
        firefox_user_agent = os.environ.get(
            "PLURIBUS_FIREFOX_USER_AGENT",
            "Mozilla/5.0 (X11; Linux x86_64; rv:146.0) Gecko/20100101 Firefox/146.0",
        )
        window_width, window_height = 1280, 720
        size_env = (os.environ.get("PLURIBUS_BROWSER_WINDOW_SIZE") or "").strip().lower()
        if size_env and "x" in size_env:
            try:
                w_raw, h_raw = size_env.split("x", 1)
                window_width, window_height = int(w_raw), int(h_raw)
            except Exception:
                pass
        else:
            try:
                window_width = int(os.environ.get("PLURIBUS_BROWSER_WINDOW_WIDTH") or window_width)
                window_height = int(os.environ.get("PLURIBUS_BROWSER_WINDOW_HEIGHT") or window_height)
            except Exception:
                window_width, window_height = 1280, 720

        split_mode = (os.environ.get("PLURIBUS_BROWSER_SPLIT_MODE") or "").strip().lower()
        split_ratio_env = (os.environ.get("PLURIBUS_BROWSER_SPLIT_RATIO") or "0.5").strip()
        split_primary = (os.environ.get("PLURIBUS_BROWSER_SPLIT_PRIMARY") or "firefox").strip().lower()
        split_layout: dict[str, tuple[int, int, int, int]] | None = None
        if split_primary not in {"firefox", "chromium"}:
            split_primary = "firefox"
        if not headless and display and {"chromium", "firefox"} <= needed_browsers:
            if not split_mode:
                split_mode = "vertical"
            if split_mode in {"vertical", "horizontal"}:
                try:
                    split_ratio = float(split_ratio_env)
                except Exception:
                    split_ratio = 0.5
                screen_size = detect_screen_size(display) or (window_width, window_height)
                split_layout = _compute_split_layout(
                    screen_size[0],
                    screen_size[1],
                    split_mode,
                    split_ratio,
                    split_primary,
                )

        launch_errors: dict[str, str] = {}
        for btype in needed_browsers:
            print(f"  Launching {btype} context...")
            b_data_dir = self.user_data_dir_root / btype
            b_data_dir.mkdir(parents=True, exist_ok=True)
            b_layout = split_layout.get(btype) if split_layout else None

            launch_args = [
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ]

            browser_env = dict(os.environ)
            if display:
                browser_env["DISPLAY"] = display
                if not browser_env.get("XAUTHORITY"):
                    xauth_path = Path.home() / ".Xauthority"
                    if xauth_path.exists():
                        browser_env["XAUTHORITY"] = str(xauth_path)

            # Robotic mode check: Only disable AutomationControlled for Chromium (Claude/ChatGPT).
            # The user noted that disabling this breaks Gemini.
            if btype == "chromium":
                launch_args.append("--disable-blink-features=AutomationControlled")

            if not headless and display:
                if b_layout:
                    _, _, b_width, b_height = b_layout
                    launch_args.append(f"--window-size={b_width},{b_height}")
                    if btype == "chromium":
                        b_x, b_y, _, _ = b_layout
                        launch_args.append(f"--window-position={b_x},{b_y}")
                else:
                    launch_args.extend([
                        "--start-maximized",
                        f"--window-size={window_width},{window_height}",
                    ])

            browser_exec = None
            try:
                if btype == "chromium":
                    browser_exec = os.environ.get("PLURIBUS_BROWSER_EXEC_CHROMIUM", "").strip() or os.environ.get("PLURIBUS_BROWSER_EXEC", "").strip() or None
                    if not browser_exec:
                        for candidate in ("/usr/bin/google-chrome-stable", "/usr/bin/google-chrome", "/opt/google/chrome/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"):
                            if Path(candidate).exists():
                                browser_exec = candidate
                                break

                    # Pre-launch: ensure single browser instance
                    if _guardian:
                        _guardian.ensure_single_browser()

                    self.contexts[btype] = await self.playwright.chromium.launch_persistent_context(
                        str(b_data_dir),
                        headless=headless,
                        viewport={"width": 1280, "height": 800}
                        if headless
                        else {
                            "width": (b_layout[2] if b_layout else window_width),
                            "height": (b_layout[3] if b_layout else window_height),
                        },
                        user_agent=chromium_user_agent,
                        locale="en-US",
                        args=launch_args,
                        env=browser_env,
                        executable_path=browser_exec,
                    )
                elif btype == "firefox":
                    browser_exec = os.environ.get("PLURIBUS_BROWSER_EXEC_FIREFOX", "").strip() or None
                    if not browser_exec:
                        for candidate in ("/usr/bin/firefox", "/usr/bin/firefox-esr"):
                            if Path(candidate).exists():
                                browser_exec = candidate
                                break

                    firefox_prefs = {
                        "dom.webdriver.enabled": False,
                        "dom.webdriver.testing.enabled": False,
                        "general.useragent.override": firefox_user_agent,
                        # Sandbox disables for restricted VPS kernels (avoid user-namespace EPERM).
                        "security.sandbox.content.level": 0,
                        "security.sandbox.gpu.level": 0,
                        "security.sandbox.rdd.level": 0,
                        "security.sandbox.socket.process.level": 0,
                    }
                    self.contexts[btype] = await self.playwright.firefox.launch_persistent_context(
                        str(b_data_dir),
                        headless=headless,
                        viewport={"width": 1280, "height": 800}
                        if headless
                        else {
                            "width": (b_layout[2] if b_layout else window_width),
                            "height": (b_layout[3] if b_layout else window_height),
                        },
                        user_agent=firefox_user_agent,
                        locale="en-US",
                        args=launch_args,
                        env=browser_env,
                        executable_path=browser_exec,
                        firefox_user_prefs=firefox_prefs,
                    )
            except Exception as e:
                launch_errors[btype] = str(e)
                print(f"  ERROR: Failed to launch {btype} context: {e}")
                continue

            # Track PID
            b_pid = _find_browser_pid(b_data_dir)
            self.state.browser_pids[btype] = b_pid

            # Apply stealth/init scripts
            if btype == "chromium":
                try:
                    await self.contexts[btype].add_init_script(
                        """
                        () => {
                          Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                          Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                          Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                          window.chrome = window.chrome || { runtime: {} };
                        }
                        """
                    )
                except Exception:
                    pass
            elif btype == "firefox":
                try:
                    await self.contexts[btype].add_init_script(
                        """
                        () => {
                          Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                          Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                          Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
                        }
                        """
                    )
                except Exception:
                    pass

        if not self.contexts:
            raise RuntimeError(f"No browser contexts launched: {launch_errors}")

        if split_layout and display and not headless:
            await asyncio.sleep(1.0)
            moved = await asyncio.to_thread(_apply_split_layout, display, split_layout)
            if not moved:
                print("  [layout] Split layout requested, but window move was unavailable.")

        # Update state
        self.state.running = True
        self.state.pid = os.getpid()
        self.state.started_at = now_iso()
        self.save_state()

        # Emit startup event
        startup_data = {
            "pid": self.state.pid,
            "browser_pids": self.state.browser_pids,
            "headless": headless,
            "vnc_mode": {
                "enabled": vnc_mode_enabled,
                "display": display,
                "vnc_detected": vnc_server_running,
            },
        }
        if vnc_mode_enabled and display:
            startup_data["vnc_connection"] = get_vnc_connection_info()

        append_bus_event(self.bus_dir, "browser.daemon.started", startup_data)

        # Emit VNC mode status event if enabled
        if vnc_mode_enabled:
            append_bus_event(self.bus_dir, "browser.vnc.mode_enabled", {
                "display": display,
                "vnc_detected": vnc_server_running,
                "connection_info": get_vnc_connection_info() if display else None,
            })

        print(f"Browser daemon started (PID {self.state.pid})")
        if vnc_mode_enabled:
            print(f"  Mode: VNC (visible browser)")
            if display:
                vnc_info = get_vnc_connection_info()
                print(f"  Connect via VNC: {vnc_info['connection_string']}")
        else:
            print(f"  Mode: {'headless' if headless else 'visible'}")

        # Cleanup old snapshots on startup
        await self.cleanup_old_snapshots()

        # Initialize tabs for each provider
        await self.init_tabs()

        # Run loops until shutdown
        tasks = [
            asyncio.create_task(self.run_health_loop(), name="browser_health_loop"),
            asyncio.create_task(self.run_bus_loop(), name="browser_bus_loop"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await self.stop()

    async def _maybe_clear_bot_challenge(self, page: Any, tab: TabSession, *, wait_s: int = 20) -> bool:
        """If we're on a transient bot-challenge page, give it a brief chance to clear."""
        if not tab.title or "just a moment" not in tab.title.lower():
            return True
        deadline = time.time() + max(1, int(wait_s))
        while time.time() < deadline:
            await asyncio.sleep(1)
            try:
                title = await page.title()
            except Exception:
                title = ""
            if title and "just a moment" not in title.lower():
                tab.title = title
                try:
                    tab.current_url = page.url
                except Exception:
                    pass
                return True
        return False

    async def _chatgpt_noauth_present(self, page: Any) -> bool:
        try:
            html = await page.content()
        except Exception:
            html = ""
        return _chatgpt_noauth_gate(html)

    async def _maybe_dismiss_chatgpt_noauth(self, page: Any) -> bool:
        """
        ChatGPT can show a "Thanks for trying ChatGPT" modal with a "Stay logged out" link.
        If possible, dismiss it once; otherwise, treat as needs_login.
        """
        if not await self._chatgpt_noauth_present(page):
            return True
        # Try a couple of common locators; best-effort only.
        for _ in range(2):
            try:
                await page.get_by_role("link", name="Stay logged out").click(timeout=3000)
                await asyncio.sleep(1)
            except Exception:
                try:
                    await page.get_by_text("Stay logged out").click(timeout=3000)
                    await asyncio.sleep(1)
                except Exception:
                    break
            if not await self._chatgpt_noauth_present(page):
                return True
        return not await self._chatgpt_noauth_present(page)

    async def _try_gemini_onboarding(self, page: Any, tab: TabSession | None) -> bool:
        """Best-effort Gemini /welcome click-through to exit onboarding."""
        try:
            url = page.url.lower()
        except Exception:
            url = ""
        if "/welcome" not in url:
            return False
        # Try common CTAs
        for selector in [
            "button:has-text(\"Get started\")",
            "button:has-text(\"Continue\")",
            "button:has-text(\"Build\")",
            "button:has-text(\"Start\")",
            "text=Get started",
            "text=Continue",
            "text=Build",
            "text=Start",
        ]:
            try:
                await page.click(selector, timeout=4000)
                await asyncio.sleep(1)
                await page.wait_for_load_state("domcontentloaded", timeout=10_000)
                current = page.url.lower()
                if "/welcome" not in current:
                    if tab:
                        tab.status = "ready"
                        tab.error = None
                        tab.current_url = page.url
                        tab.title = await page.title()
                        self.save_state()
                    return True
            except Exception:
                continue
        return False

    async def _try_gemini_project_picker(self, page: Any, tab: TabSession | None) -> bool:
        """Best-effort Gemini project/prompt picker navigation.

        When landing on AI Studio after login, user may see a project picker or
        model selector. This function tries to click through to the main chat interface.
        Returns True if successfully navigated past project picker.
        """
        try:
            url = page.url.lower()
        except Exception:
            url = ""

        # Common project picker URL patterns
        is_project_picker = any(p in url for p in ["/prompts", "/projects", "/app/prompts"])
        is_ready = "/app" in url and "/prompts" not in url and "/welcome" not in url

        if is_ready:
            return True

        if not is_project_picker and "/welcome" not in url:
            # Not on a known project picker, check if there's a prompt input (ready state)
            try:
                prompt_input = await page.query_selector('textarea[aria-label*="prompt" i], textarea[placeholder*="prompt" i], textarea')
                if prompt_input:
                    return True
            except Exception:
                pass

        # Try to find and click "New chat", "Create", or first available project
        selectors_to_try = [
            'button:has-text("New chat")',
            'button:has-text("Create")',
            'button:has-text("New prompt")',
            'a:has-text("New chat")',
            '[data-testid="new-chat"]',
            '.project-card:first-child',  # First project in a grid
            '[role="listitem"]:first-child button',  # First item in a list
            'button:has-text("Start chatting")',
        ]

        for selector in selectors_to_try:
            try:
                el = page.locator(selector).first
                if await el.is_visible(timeout=2000):
                    await el.click(timeout=5000)
                    await asyncio.sleep(1.5)
                    await page.wait_for_load_state("domcontentloaded", timeout=10_000)
                    new_url = page.url.lower()
                    # Check if we're now in a chat-ready state
                    if "/app" in new_url and "/prompts" not in new_url and "/welcome" not in new_url:
                        if tab:
                            tab.status = "ready"
                            tab.error = None
                            tab.current_url = page.url
                            try:
                                tab.title = await page.title()
                            except Exception:
                                pass
                            self.save_state()
                        append_bus_event(self.bus_dir, "browser.vnc.project_selected", {
                            "provider": "gemini-web",
                            "url": page.url,
                        })
                        return True
            except Exception:
                continue

        # If we're still on project picker, mark as needs_project
        if tab and is_project_picker:
            tab.status = "needs_project"
            tab.error = "Project selection required"
            tab.current_url = page.url
            self.save_state()
            append_bus_event(self.bus_dir, "browser.vnc.needs_project", {
                "provider": "gemini-web",
                "url": page.url,
                "message": "Auto-selection failed; manual selection may be needed"
            })

        return False

    async def init_tabs(self) -> None:
        """Initialize browser tabs for each provider."""
        for provider_id, config in WEB_PROVIDERS.items():
            try:
                btype = config.get("browser_type", "chromium")
                context = self.contexts.get(btype)
                if not context:
                    if self.contexts:
                        fallback_type = next(iter(self.contexts.keys()))
                        context = self.contexts[fallback_type]
                        print(f"  WARN: Context for {btype} not available; using {fallback_type} for {provider_id}.")
                    else:
                        print(f"  ERROR: Context for {btype} not available. Skipping {provider_id}.")
                        continue

                print(f"  Initializing tab for {config['name']} ({btype})...")
                page = await context.new_page()

                # Apply playwright-stealth for better bot evasion (if enabled for this provider)
                if config.get("stealth") and STEALTH_AVAILABLE and STEALTH:
                    try:
                        await STEALTH.apply_stealth_async(page)
                        print(f"    [stealth] Applied to {provider_id}")
                    except Exception as e:
                        print(f"    [stealth] Failed for {provider_id}: {e}")

                self.pages[provider_id] = page

                # Navigate to provider URL
                await page.goto(config["url"], wait_until="domcontentloaded", timeout=30000)

                # Create tab session
                tab = TabSession(
                    provider_id=provider_id,
                    tab_id=str(uuid.uuid4())[:8],
                    url=config["url"],
                    status="ready",
                    session_start=now_iso(),
                    last_health_check=now_iso(),
                    last_activity=now_iso(),
                )
                self.state.tabs[provider_id] = tab

                # Check if login required
                current_url = page.url
                tab.current_url = current_url
                try:
                    tab.title = await page.title()
                except Exception:
                    tab.title = ""

                # Bot challenge detection (common on claude.ai / chatgpt.com in headless mode)
                if tab.title and "just a moment" in tab.title.lower():
                    if await self._maybe_clear_bot_challenge(page, tab, wait_s=20):
                        tab.status = "ready"
                        tab.error = None
                        print(f"    {config['name']}: ready")
                        self.save_state()
                        continue
                    tab.status = "blocked_bot"
                    tab.error = f"Bot challenge detected at {current_url}"
                    print(f"    {config['name']}: blocked (bot challenge)")
                # Gemini AI Studio welcome/onboarding screen (no prompt box until click-through)
                elif "/welcome" in current_url:
                    print(f"    {config['name']}: attempting auto-onboarding...")
                    # AUTO-ONBOARDING: Try to click through welcome screen
                    onboarding_clicked = False
                    for btn_selector in [
                        'button:has-text("Get started")',
                        'button:has-text("Build")',
                        'button:has-text("Continue")',
                        'button:has-text("Start")',
                        '[data-testid="get-started"]',
                        'a:has-text("Get started")',
                    ]:
                        try:
                            btn = page.locator(btn_selector).first
                            if await btn.is_visible(timeout=2000):
                                await btn.click(timeout=5000)
                                onboarding_clicked = True
                                await asyncio.sleep(2)
                                break
                        except Exception:
                            continue
                    # Fallback: try helper
                    if not onboarding_clicked:
                        onboarding_clicked = await self._try_gemini_onboarding(page, tab)

                    # Re-check URL
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=10_000)
                        current_url = page.url
                        tab.current_url = current_url
                    except Exception:
                        pass

                    if "/welcome" not in current_url:
                        tab.status = "ready"
                        tab.error = None
                        print(f"    {config['name']}: ready (auto-onboarding successful)")
                        append_bus_event(self.bus_dir, "browser.vnc.onboarding_complete", {
                            "provider": provider_id,
                            "url": current_url,
                        })
                    else:
                        tab.status = "needs_onboarding"
                        tab.error = f"Onboarding required at {current_url}"
                        print(f"    {config['name']}: needs onboarding (auto-click {'attempted' if onboarding_clicked else 'failed'})")
                # ChatGPT can appear "ready" while showing a logged-out gate modal.
                elif provider_id == "chatgpt-web":
                    try:
                        html_content = await page.content()
                    except Exception:
                        html_content = ""
                    if _chatgpt_noauth_gate(html_content):
                        tab.status = "needs_login"
                        tab.error = f"Login required at {current_url}"
                        print(f"    {config['name']}: needs login")
                if "login" in current_url.lower() or "auth" in current_url.lower() or "accounts" in current_url.lower():
                    tab.status = "needs_login"
                    tab.error = f"Login required at {current_url}"
                    print(f"    {config['name']}: needs login")
                elif tab.status == "ready":
                    print(f"    {config['name']}: ready")

                self.save_state()

                append_bus_event(self.bus_dir, "browser.tab.initialized", {
                    "provider": provider_id,
                    "status": tab.status,
                    "url": current_url,
                    "title": tab.title,
                })

            except Exception as e:
                print(f"    {config['name']}: error - {e}")
                # Capture snapshot on initialization error
                await self.save_snapshot(page, provider_id, "init_failure", str(e))
                self.state.tabs[provider_id] = TabSession(
                    provider_id=provider_id,
                    tab_id=str(uuid.uuid4())[:8],
                    url=config["url"],
                    status="error",
                    error=str(e),
                    session_start=now_iso(),
                )
                self.save_state()

        # Initialize dashboard tab for CUA monitoring
        await self.init_dashboard_tab()

    async def init_dashboard_tab(self) -> None:
        """Initialize dashboard tab for CUA auto-reload on errors."""
        try:
            dashboard_url = DASHBOARD_CONFIG["url"]
            print(f"  Initializing dashboard tab ({dashboard_url})...")
            # Default dashboard to chromium context if available
            context = self.contexts.get("chromium") or next(iter(self.contexts.values()))
            self.dashboard_page = await context.new_page()
            await self.dashboard_page.goto(dashboard_url, wait_until="domcontentloaded", timeout=30000)
            append_bus_event(self.bus_dir, "browser.dashboard.initialized", {
                "url": dashboard_url,
                "status": "ready",
            })
            print(f"    Dashboard: ready")
        except Exception as e:
            print(f"    Dashboard: error - {e}")
            append_bus_event(self.bus_dir, "browser.dashboard.error", {"error": str(e)[:200]})

    # ========================================================================
    # VNC Mode Control Methods
    # ========================================================================

    async def enable_vnc_mode(self) -> dict:
        """Enable VNC mode for manual OAuth login.

        Returns status dict with success/failure and connection info.
        """
        display = detect_display()
        vnc_running = is_vnc_server_running()

        if not display:
            return {
                "success": False,
                "error": "No DISPLAY environment variable set. Start a VNC server first.",
                "hint": "Run: vncserver :1 or x11vnc -display :0"
            }

        # Update state
        self.state.vnc_mode.enabled = True
        self.state.vnc_mode.display = display
        self.state.vnc_mode.vnc_detected = vnc_running
        self.state.vnc_mode.started_at = now_iso()

        # Identify providers that need login
        pending = []
        for provider_id, tab in self.state.tabs.items():
            if tab.status in ("needs_login", "needs_onboarding", "needs_code", "blocked_bot"):
                pending.append(provider_id)

        self.state.vnc_mode.login_providers_pending = pending
        self.state.vnc_mode.login_providers_completed = []

        self.save_state()

        # Emit bus event
        vnc_info = get_vnc_connection_info()
        append_bus_event(self.bus_dir, "browser.vnc.mode_enabled", {
            "display": display,
            "vnc_detected": vnc_running,
            "connection_info": vnc_info,
            "providers_pending": pending,
        })

        return {
            "success": True,
            "display": display,
            "vnc_detected": vnc_running,
            "connection_info": vnc_info,
            "providers_pending": pending,
            "message": f"VNC mode enabled. Connect to {vnc_info['connection_string']} to complete login."
        }

    async def disable_vnc_mode(self) -> dict:
        """Disable VNC mode after manual login is complete."""
        was_enabled = self.state.vnc_mode.enabled

        # Check which providers were successfully logged in
        completed = []
        still_pending = []

        for provider_id in self.state.vnc_mode.login_providers_pending:
            tab = self.state.tabs.get(provider_id)
            if tab and tab.status == "ready":
                completed.append(provider_id)
            elif tab:
                still_pending.append(provider_id)

        self.state.vnc_mode.login_providers_completed = completed
        self.state.vnc_mode.login_providers_pending = still_pending
        self.state.vnc_mode.enabled = False

        self.save_state()

        # Emit bus event
        append_bus_event(self.bus_dir, "browser.vnc.mode_disabled", {
            "was_enabled": was_enabled,
            "providers_completed": completed,
            "providers_still_pending": still_pending,
        })

        return {
            "success": True,
            "was_enabled": was_enabled,
            "providers_completed": completed,
            "providers_still_pending": still_pending,
            "message": f"VNC mode disabled. {len(completed)} provider(s) logged in successfully."
        }

    def get_vnc_status(self) -> dict:
        """Get current VNC mode status."""
        display = detect_display()
        vnc_running = is_vnc_server_running()

        # Check current login status of providers
        providers_status = {}
        for provider_id, tab in self.state.tabs.items():
            providers_status[provider_id] = {
                "status": tab.status,
                "needs_action": tab.status in ("needs_login", "needs_onboarding", "needs_code", "blocked_bot"),
                "url": tab.current_url,
            }

        result = {
            "enabled": self.state.vnc_mode.enabled,
            "display": display,
            "vnc_detected": vnc_running,
            "started_at": self.state.vnc_mode.started_at,
            "providers_pending": self.state.vnc_mode.login_providers_pending,
            "providers_completed": self.state.vnc_mode.login_providers_completed,
            "providers_status": providers_status,
        }

        if display:
            result["connection_info"] = get_vnc_connection_info()

        return result

    async def navigate_to_login(self, provider_id: str, *, req_id: str | None = None) -> dict:
        """Navigate a provider tab to its login page for manual authentication.

        Useful when VNC mode is enabled and user needs to log in.
        """
        config = WEB_PROVIDERS.get(provider_id)
        if not config:
            return {"success": False, "error": f"Unknown provider: {provider_id}"}

        page = self.pages.get(provider_id)
        if not page:
            return {"success": False, "error": f"No page for provider: {provider_id}"}

        tab = self.state.tabs.get(provider_id)

        try:
            # Prefer explicit login URL when available; some providers redirect unreliably.
            login_url = config.get("login_url") or config["url"]
            if provider_id == "gemini-web":
                # Avoid landing on a generic/parameterless Google login URL; start at the provider so the
                # redirect includes the correct `continue=` flow (prevents Error 400 edge cases).
                login_url = config.get("url") or login_url

            # Ensure the target tab becomes visible in the VNC browser window.
            try:
                await page.bring_to_front()
            except Exception:
                pass

            await page.goto(login_url, wait_until="domcontentloaded", timeout=30000)

            # Wait a moment for any redirects
            await asyncio.sleep(2)

            current_url = page.url
            title = await page.title()

            if tab:
                tab.current_url = current_url
                tab.title = title
                self.save_state()

            # Emit an informational artifact (distinct from the request topic to avoid loops).
            append_bus_event(
                self.bus_dir,
                "browser.vnc.navigate_login.performed",
                {
                    "req_id": req_id,
                    "provider": provider_id,
                    "url": current_url,
                    "title": title,
                },
            )

            return {
                "success": True,
                "provider": provider_id,
                "current_url": current_url,
                "title": title,
                "message": f"Navigate to {current_url} and complete login via VNC"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def focus_tab(self, provider_id: str, *, req_id: str | None = None) -> dict:
        """Bring an existing provider tab to the front (VNC-visible)."""
        config = WEB_PROVIDERS.get(provider_id)
        if not config:
            return {"success": False, "error": f"Unknown provider: {provider_id}"}

        page = self.pages.get(provider_id)
        if not page:
            return {"success": False, "error": f"No page for provider: {provider_id}"}

        tab = self.state.tabs.get(provider_id)

        try:
            try:
                await page.bring_to_front()
            except Exception:
                pass

            await asyncio.sleep(0.1)
            current_url = page.url
            title = await page.title()

            if tab:
                tab.current_url = current_url
                tab.title = title
                self.save_state()

            append_bus_event(
                self.bus_dir,
                "browser.vnc.focus_tab.performed",
                {
                    "req_id": req_id,
                    "provider": provider_id,
                    "url": current_url,
                    "title": title,
                },
            )

            return {
                "success": True,
                "provider": provider_id,
                "current_url": current_url,
                "title": title,
                "message": f"Focused {provider_id}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def check_login_complete(self, provider_id: str) -> dict:
        """Check if manual login has been completed for a provider.

        Called after user completes OAuth login via VNC.
        """
        config = WEB_PROVIDERS.get(provider_id)
        if not config:
            return {"success": False, "error": f"Unknown provider: {provider_id}"}

        page = self.pages.get(provider_id)
        if not page:
            return {"success": False, "error": f"No page for provider: {provider_id}"}

        tab = self.state.tabs.get(provider_id)

        try:
            current_url = page.url
            title = await page.title()

            if tab:
                tab.current_url = current_url
                tab.title = title

            # Check if still on login page
            login_url = config.get("login_url", "")
            is_on_login = _looks_like_login_flow(current_url, login_url=login_url)

            # Check for bot challenge
            is_bot_blocked = title and "just a moment" in title.lower()

            # Check for ChatGPT noauth gate
            is_chatgpt_noauth = False
            if provider_id == "chatgpt-web":
                is_chatgpt_noauth = await self._chatgpt_noauth_present(page)

            onboarding_required = provider_id == "gemini-web" and ("/welcome" in (current_url or "").lower())
            login_complete = not is_on_login and not is_bot_blocked and not is_chatgpt_noauth

            if login_complete:
                if onboarding_required:
                    # AUTO-ONBOARDING: Click through the welcome/onboarding screen
                    append_bus_event(self.bus_dir, "browser.vnc.onboarding_starting", {
                        "provider": provider_id,
                        "url": current_url,
                        "message": "Attempting auto-onboarding click-through"
                    })

                    onboarding_clicked = False
                    # Try common CTA buttons in order of likelihood
                    for btn_selector in [
                        'button:has-text("Get started")',
                        'button:has-text("Build")',
                        'button:has-text("Continue")',
                        'button:has-text("Start")',
                        '[data-testid="get-started"]',
                        'a:has-text("Get started")',
                    ]:
                        try:
                            btn = page.locator(btn_selector).first
                            if await btn.is_visible(timeout=2000):
                                await btn.click(timeout=5000)
                                onboarding_clicked = True
                                await asyncio.sleep(2)  # Wait for navigation
                                break
                        except Exception:
                            continue

                    # Wait for page to settle after click
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=10_000)
                    except Exception:
                        pass

                    # Re-check URL after click
                    try:
                        current_url = page.url
                        if tab:
                            tab.current_url = current_url
                    except Exception:
                        pass

                    # Check if we're still on welcome page
                    still_onboarding = "/welcome" in (current_url or "").lower()

                    if still_onboarding and not onboarding_clicked:
                        # Couldn't click through - mark as needs_onboarding
                        if tab:
                            tab.status = "needs_onboarding"
                            tab.error = f"Onboarding required at {current_url} (auto-click failed)"
                        self.save_state()
                        append_bus_event(self.bus_dir, "browser.vnc.onboarding_required", {
                            "provider": provider_id,
                            "url": current_url,
                            "auto_click_attempted": True,
                            "auto_click_success": False,
                        })
                        return {
                            "success": True,
                            "login_complete": True,
                            "onboarding_required": True,
                            "provider": provider_id,
                            "status": "needs_onboarding",
                            "current_url": current_url,
                            "title": title,
                            "message": f"Login complete for {config['name']}, but auto-onboarding failed"
                        }
                    elif still_onboarding:
                        # Clicked but still on welcome - try one more time
                        await asyncio.sleep(3)
                        try:
                            current_url = page.url
                        except Exception:
                            pass
                        still_onboarding = "/welcome" in (current_url or "").lower()

                    if not still_onboarding:
                        # Successfully passed onboarding!
                        append_bus_event(self.bus_dir, "browser.vnc.onboarding_complete", {
                            "provider": provider_id,
                            "url": current_url,
                            "message": "Auto-onboarding successful"
                        })
                        # Fall through to mark as ready below

                if tab:
                    tab.status = "ready"
                    tab.error = None

                # Update VNC mode tracking
                if provider_id in self.state.vnc_mode.login_providers_pending:
                    self.state.vnc_mode.login_providers_pending.remove(provider_id)
                if provider_id not in self.state.vnc_mode.login_providers_completed:
                    self.state.vnc_mode.login_providers_completed.append(provider_id)

                self.save_state()

                # Emit success event
                append_bus_event(self.bus_dir, "browser.vnc.login_complete", {
                    "provider": provider_id,
                    "url": current_url,
                })

                return {
                    "success": True,
                    "login_complete": True,
                    "onboarding_required": False,
                    "provider": provider_id,
                    "status": "ready",
                    "message": f"Login complete for {config['name']}"
                }
            else:
                reason = "on_login_page" if is_on_login else "bot_blocked" if is_bot_blocked else "noauth_gate" if is_chatgpt_noauth else "unknown"
                if tab:
                    tab.status = "needs_login" if is_on_login or is_chatgpt_noauth else "blocked_bot" if is_bot_blocked else tab.status
                    tab.error = f"Login not complete: {reason}"
                self.save_state()

                return {
                    "success": True,
                    "login_complete": False,
                    "onboarding_required": False,
                    "provider": provider_id,
                    "reason": reason,
                    "current_url": current_url,
                    "message": f"Login not complete for {config['name']}: {reason}"
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def inject_cli_auth(self, provider_id: str) -> dict:
        """Inject CLI OAuth credentials into browser session for a provider.

        Bridges authentication from CLI tools (Claude Code, Gemini CLI) to browser sessions.

        Returns: {"success": bool, "message": str, "method": str}
        """
        import os
        from pathlib import Path

        page = self.pages.get(provider_id)
        if not page:
            return {"success": False, "message": f"No page for provider {provider_id}", "method": "none"}

        tab = self.state.tabs.get(provider_id)
        config = WEB_PROVIDERS.get(provider_id, {})

        try:
            if provider_id == "claude-web":
                # Load Claude Code OAuth credentials
                claude_creds_paths = [
                    Path.home() / ".claude" / ".credentials.json",
                    Path("/var/lib/pluribus/.pluribus/agent_homes/claude/.claude/.credentials.json"),
                    self.root / ".pluribus" / "agent_homes" / "claude" / ".claude" / ".credentials.json",
                ]

                creds = None
                for creds_path in claude_creds_paths:
                    if creds_path.exists():
                        try:
                            creds = json.loads(creds_path.read_text())
                            break
                        except Exception:
                            continue

                if not creds or "claudeAiOauth" not in creds:
                    return {"success": False, "message": "No Claude OAuth credentials found", "method": "none"}

                oauth = creds["claudeAiOauth"]
                access_token = oauth.get("accessToken", "")
                refresh_token = oauth.get("refreshToken", "")

                if not access_token:
                    return {"success": False, "message": "Claude OAuth accessToken missing", "method": "none"}

                # Inject via localStorage (Claude web checks localStorage for auth state)
                await page.evaluate(f"""() => {{
                    try {{
                        // Store OAuth tokens in localStorage
                        localStorage.setItem('claude_access_token', '{access_token}');
                        localStorage.setItem('claude_refresh_token', '{refresh_token}');
                        localStorage.setItem('claude_auth_injected', 'true');
                        return true;
                    }} catch (e) {{
                        return false;
                    }}
                }}""")

                # Also try setting as cookie (some flows check cookies)
                await page.context.add_cookies([{
                    "name": "__claude_oauth_token",
                    "value": access_token[:100],  # Truncate for cookie safety
                    "domain": ".claude.ai",
                    "path": "/",
                    "httpOnly": False,
                    "secure": True,
                    "sameSite": "Lax",
                }])

                # Store in user_auths for future reference
                self.user_auths["claude-web"] = {
                    "token": access_token[:20] + "...",  # Don't store full token in state
                    "method": "localStorage+cookie",
                    "injected_at": now_iso(),
                }
                self.save_state()

                # Try refreshing the page to pick up the auth
                await page.reload(wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(2)

                # Check if we're past the login screen now
                current_url = page.url
                title = await page.title()

                if "login" not in current_url.lower() and "just a moment" not in title.lower():
                    if tab:
                        tab.status = "ready"
                        tab.error = None
                        tab.current_url = current_url
                        tab.title = title
                    append_bus_event(self.bus_dir, "browser.auth.injected", {
                        "provider": provider_id,
                        "method": "localStorage+cookie",
                        "success": True,
                    })
                    return {"success": True, "message": "Claude auth injected and verified", "method": "localStorage+cookie"}
                else:
                    return {"success": False, "message": f"Auth injected but still on login page: {current_url}", "method": "localStorage+cookie"}

            elif provider_id == "gemini-web":
                # For Gemini, we need Google OAuth. Check for gcloud credentials
                gcloud_creds_paths = [
                    Path.home() / ".config" / "gcloud" / "application_default_credentials.json",
                    Path("/var/lib/pluribus/.pluribus/agent_homes/gemini/.config/gcloud/application_default_credentials.json"),
                ]

                for creds_path in gcloud_creds_paths:
                    if creds_path.exists():
                        try:
                            gcloud_creds = json.loads(creds_path.read_text())
                            # Google OAuth uses different flow - need to exchange for session
                            self.user_auths["gemini-web"] = {
                                "has_gcloud": True,
                                "type": gcloud_creds.get("type", "unknown"),
                                "injected_at": now_iso(),
                            }
                            self.save_state()
                            return {"success": False, "message": "GCloud credentials found but browser OAuth exchange not implemented", "method": "gcloud_found"}
                        except Exception:
                            continue

                return {"success": False, "message": "No Google OAuth credentials found", "method": "none"}

            elif provider_id == "chatgpt-web":
                # ChatGPT uses OpenAI account - check for API key that might hint at account
                openai_key = os.environ.get("OPENAI_API_KEY", "")
                if openai_key:
                    self.user_auths["chatgpt-web"] = {
                        "has_api_key": True,
                        "injected_at": now_iso(),
                    }
                    self.save_state()
                    return {"success": False, "message": "OpenAI API key found but web session exchange not implemented", "method": "api_key_found"}
                return {"success": False, "message": "No OpenAI credentials found", "method": "none"}

            else:
                return {"success": False, "message": f"Unknown provider: {provider_id}", "method": "none"}

        except Exception as e:
            append_bus_event(self.bus_dir, "browser.auth.inject_error", {
                "provider": provider_id,
                "error": str(e)[:200],
            })
            return {"success": False, "message": str(e), "method": "error"}

    def _should_auto_login(self, provider_id: str) -> bool:
        """Throttle auto-login attempts per provider."""
        if not AUTOLOGIN_ENABLED:
            return False
        if provider_id in AUTOLOGIN_DISABLED_PROVIDERS:
            return False
        user, pw = _env_creds_for_provider(provider_id)
        if not user or not pw:
            return False
        last = self.last_auto_login_attempt.get(provider_id, 0)
        return (time.time() - last) > AUTOLOGIN_COOLDOWN_S

    async def _auto_login_provider(self, provider_id: str) -> dict:
        """
        Best-effort auto-login using env credentials (never persisted).

        Returns summary dict (success, status, message, needs_code flag).
        """
        self.last_auto_login_attempt[provider_id] = time.time()
        user, pw = _env_creds_for_provider(provider_id)
        if not user or not pw:
            return {"success": False, "status": "needs_login", "message": "no creds in env"}

        page = self.pages.get(provider_id)
        tab = self.state.tabs.get(provider_id)
        config = WEB_PROVIDERS.get(provider_id, {})
        if not page or not tab or not config:
            return {"success": False, "status": "error", "message": "page/tab missing"}

        try:
            login_url = config.get("login_url") or config.get("url")
            start_url = login_url
            if provider_id == "gemini-web":
                # Avoid navigating directly to a bare accounts.google.com URL (can yield Error 400 without flow params).
                # Instead, start at the provider URL and let it redirect into the correct Google login flow.
                start_url = config.get("url") or login_url
            try:
                await page.bring_to_front()
            except Exception:
                pass
            if provider_id != "gemini-web":
                await page.goto(start_url, wait_until="domcontentloaded", timeout=30_000)
                await asyncio.sleep(1)

            # Common selectors per provider
            email_selectors = [
                "input[type='email']",
                "input[name='email']",
                "input[name='username']",
                "input[id='username']",
            ]
            password_selectors = [
                "input[type='password']",
                "input[name='password']",
                "input[id='password']",
                "input[name='pass']",
            ]
            next_buttons = [
                "button:has-text(\"Next\")",
                "button:has-text(\"Continue\")",
                "button:has-text(\"Log in\")",
                "button:has-text(\"Sign in\")",
                "button[type='submit']",
            ]

            async def _fill_first(selectors: list[str], value: str) -> bool:
                for sel in selectors:
                    try:
                        el = await page.wait_for_selector(sel, timeout=5000)
                        await el.fill(value)
                        return True
                    except Exception:
                        continue
                return False

            async def _click_first(selectors: list[str]) -> bool:
                for sel in selectors:
                    try:
                        el = await page.wait_for_selector(sel, timeout=5000)
                        await el.click()
                        return True
                    except Exception:
                        continue
                return False

            if provider_id == "gemini-web":
                # Use Google-specific flow
                gflow = await _google_login_flow(page, user, pw, start_url=start_url)
                if gflow.get("blocked_insecure"):
                    tab.status = "blocked_bot"
                    tab.error = "Google blocked this browser: 'This browser or app may not be secure'"
                    tab.current_url = page.url
                    self.save_state()
                    append_bus_event(self.bus_dir, "browser.auto_login.blocked_insecure", {
                        "provider": provider_id,
                        "url": page.url,
                    })
                    return {"success": False, "status": "blocked_bot", "message": "insecure browser gate", "needs_code": False}
                if gflow.get("needs_code"):
                    sol = await self.solve_challenge(page, provider_id, reason="otp")
                    if sol.get("applied"):
                        # Recheck post-OTP
                        await asyncio.sleep(2)
                # continue to final checks below
            else:
                # Generic form flow
                await _fill_first(email_selectors, user)
                await _click_first(next_buttons)
                await asyncio.sleep(1.5)
                await _fill_first(password_selectors, pw)
                await _click_first(next_buttons)
                await asyncio.sleep(2.5)

            # Quick heuristic for OTP / verification challenge
            page_text = ""
            try:
                page_text = (await page.text_content("body")) or ""
            except Exception:
                page_text = ""
            needs_code = any(
                kw in page_text.lower()
                for kw in ["verification code", "2-step", "2 step", "two-factor", "otp", "enter code"]
            )
            if needs_code:
                # Try auto-fill TOTP if secret is available
                totp_secret = _get_totp_secret_for_provider(provider_id)
                totp_auto_filled = False
                if totp_secret:
                    totp_code = _generate_totp(totp_secret)
                    if totp_code:
                        otp_selectors = [
                            'input[name="totpPin"]',
                            'input[type="tel"]',
                            'input[aria-label*="code"]',
                            'input[aria-label*="verification"]',
                            'input[autocomplete="one-time-code"]',
                            'input[name="otp"]',
                            'input[id*="otp"]',
                        ]
                        for sel in otp_selectors:
                            try:
                                otp_el = await page.wait_for_selector(sel, timeout=3000)
                                await otp_el.fill(totp_code)
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(2.5)
                                totp_auto_filled = True
                                needs_code = False
                                append_bus_event(self.bus_dir, "browser.auto_login.totp_filled", {
                                    "provider": provider_id,
                                    "url": page.url,
                                })
                                break
                            except Exception:
                                continue
                if needs_code:
                    # TOTP not available or failed, fall back to VNC mode
                    if tab:
                        tab.status = "needs_code"
                        tab.error = "2FA/verification required"
                        tab.current_url = page.url
                    self.save_state()
                    append_bus_event(self.bus_dir, "browser.auto_login.needs_code", {
                        "provider": provider_id,
                        "url": page.url,
                    })
                    # Best-effort: ensure VNC mode is enabled and the tab is visible for the operator.
                    try:
                        await self.enable_vnc_mode()
                    except Exception:
                        pass
                    try:
                        await self.focus_tab(provider_id)
                    except Exception:
                        pass
                    return {"success": False, "status": "needs_code", "message": "verification code required", "needs_code": True}

            # Final state check
            current_url = page.url
            current_url_lower = (current_url or "").lower()
            title = ""
            try:
                title = await page.title()
            except Exception:
                pass
            is_on_login = _looks_like_login_flow(current_url, login_url=config.get("login_url"))
            is_bot_blocked = title and "just a moment" in title.lower()
            is_chatgpt_noauth = provider_id == "chatgpt-web" and await self._chatgpt_noauth_present(page)
            if provider_id == "gemini-web" and "/welcome" in current_url_lower:
                # /welcome is a special case: we must not mark the tab "ready" unless we exit onboarding
                # AND are not redirected into an auth flow.
                try:
                    await self._try_gemini_onboarding(page, tab)
                except Exception:
                    pass

                # Refresh classification after onboarding attempt (the click can redirect).
                await asyncio.sleep(1)
                current_url = page.url
                current_url_lower = (current_url or "").lower()
                try:
                    title = await page.title()
                except Exception:
                    pass
                is_on_login = _looks_like_login_flow(current_url, login_url=config.get("login_url"))

                # Still on welcome: keep it explicit so the dashboard can prompt the operator.
                if "/welcome" in current_url_lower:
                    tab.status = "needs_onboarding"
                    tab.error = f"Onboarding required at {current_url}"
                    tab.current_url = current_url
                    tab.title = title
                    self.save_state()
                    return {"success": False, "status": "needs_onboarding", "message": tab.error}

            # After onboarding check, also try project picker for Gemini
            if provider_id == "gemini-web" and not is_on_login:
                # Check if we're on a project picker page
                if any(p in current_url_lower for p in ["/prompts", "/projects", "/app/prompts"]):
                    if await self._try_gemini_project_picker(page, tab):
                        current_url = page.url
                        current_url_lower = (current_url or "").lower()
                        try:
                            title = await page.title()
                        except Exception:
                            pass
                        is_on_login = _looks_like_login_flow(current_url, login_url=config.get("login_url"))

            if is_bot_blocked:
                tab.status = "blocked_bot"
                tab.error = f"Bot challenge at {current_url}"
                self.save_state()
                # Try solver if enabled
                if SOLVER_ENABLED and SOLVER_CMD:
                    sol = await self.solve_challenge(page, provider_id, reason="bot_blocked")
                    if sol.get("applied"):
                        tab.status = "ready"
                        tab.error = None
                        tab.current_url = page.url
                        self.save_state()
                        return {"success": True, "status": "ready", "message": "solver cleared bot challenge"}
                return {"success": False, "status": "blocked_bot", "message": tab.error}
            if is_on_login or is_chatgpt_noauth:
                tab.status = "needs_login"
                tab.error = f"Login still required at {current_url}"
                self.save_state()
                return {"success": False, "status": "needs_login", "message": tab.error}

            tab.status = "ready"
            tab.error = None
            tab.current_url = current_url
            tab.title = title
            self.save_state()

            append_bus_event(self.bus_dir, "browser.auto_login.success", {
                "provider": provider_id,
                "url": current_url,
            })
            return {"success": True, "status": "ready", "message": "login completed"}
        except Exception as e:
            if tab:
                tab.status = "error"
                tab.error = str(e)
                self.save_state()
            append_bus_event(self.bus_dir, "browser.auto_login.error", {
                "provider": provider_id,
                "error": str(e)[:200],
            })
            return {"success": False, "status": "error", "message": str(e)}

    async def reload_dashboard(self, reason: str) -> None:
        """Reload dashboard tab (CUA action)."""
        now = time.time()
        cooldown = DASHBOARD_CONFIG["reload_cooldown_s"]
        if (now - self.last_dashboard_reload) < cooldown:
            return  # Still in cooldown

        self.last_dashboard_reload = now
        try:
            if self.dashboard_page:
                await self.dashboard_page.reload(wait_until="domcontentloaded", timeout=30000)
                append_bus_event(self.bus_dir, "browser.dashboard.reloaded", {
                    "reason": reason,
                    "url": DASHBOARD_CONFIG["url"],
                })
                print(f"[CUA] Dashboard reloaded: {reason}")
        except Exception as e:
            append_bus_event(self.bus_dir, "browser.dashboard.reload_error", {"error": str(e)[:200]})

    def _count_recent_errors(self) -> int:
        """Count telemetry errors in sliding window."""
        now = time.time()
        window = DASHBOARD_CONFIG["error_window_s"]
        self.dashboard_errors = [t for t in self.dashboard_errors if (now - t) < window]
        return len(self.dashboard_errors)

    async def run_health_loop(self) -> None:
        """Run periodic health checks."""
        while self._running:
            try:
                # Sleep in short steps so stop() takes effect promptly.
                for _ in range(60):
                    if not self._running:
                        break
                    await asyncio.sleep(1)
                if not self._running:
                    break
                await self.health_check()
                await self.maybe_auto_login()
                await self.cleanup_old_snapshots()  # Cleanup after health check
                await self.watchdog_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health check error: {e}")

    def _load_cursor(self) -> int:
        try:
            if self.cursor_path.exists():
                raw = self.cursor_path.read_text().strip()
                return int(raw or "0")
        except Exception:
            pass
        return 0

    def _save_cursor(self, pos: int) -> None:
        try:
            self.cursor_path.parent.mkdir(parents=True, exist_ok=True)
            self.cursor_path.write_text(str(int(pos)))
        except Exception:
            pass

    async def run_bus_loop(self) -> None:
        """Consume browser inference requests from the append-only bus."""
        pos = self._load_cursor()
        while self._running:
            try:
                await asyncio.sleep(0.2)
                if not self.events_path.exists():
                    continue

                with self.events_path.open("rb") as f:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    except Exception:
                        pass
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass

                if not chunk:
                    continue

                for raw in chunk.splitlines():
                    try:
                        ev = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue

                    topic = ev.get("topic") or ""
                    data = ev.get("data") or {}
                    if not isinstance(data, dict):
                        data = {}

                    # === CUA: Telemetry error tracking ===
                    if topic.startswith("telemetry.client.") and ev.get("level") == "error":
                        self.dashboard_errors.append(time.time())
                        error_count = self._count_recent_errors()
                        if error_count >= DASHBOARD_CONFIG["error_threshold"]:
                            await self.reload_dashboard(f"error_spike:{error_count}")
                            self.dashboard_errors.clear()  # Reset after reload
                        continue

                    # === CUA: Explicit dashboard reload request ===
                    if topic == "browser.dashboard.reload":
                        reason = str(data.get("reason") or "explicit_request")
                        await self.reload_dashboard(reason)
                        continue

                    # === CUA: OITERATE action triggered ===
                    if topic == "oiterate.action_triggered":
                        action = str(data.get("action") or "")
                        if action == "iterate" and data.get("agenda"):
                            # Could trigger browser actions based on agenda frames
                            append_bus_event(self.bus_dir, "browser.cua.oiterate_ack", {
                                "session_id": data.get("session_id"),
                                "agenda_count": len(data.get("agenda", [])),
                            })
                        continue

                    # === CUA: Inject CLI OAuth credentials into browser ===
                    if topic == "browser.inject_auth":
                        provider_id = str(data.get("provider") or "")
                        req_id = str(data.get("req_id") or "")
                        if provider_id:
                            result = await self.inject_cli_auth(provider_id)
                            append_bus_event(self.bus_dir, "browser.inject_auth.response", {
                                "req_id": req_id,
                                "provider": provider_id,
                                **result,
                            })
                        continue

                    # === VNC Mode: Enable VNC mode for manual login ===
                    if topic == "browser.vnc.enable":
                        req_id = str(data.get("req_id") or "")
                        result = await self.enable_vnc_mode()
                        append_bus_event(self.bus_dir, "browser.vnc.enable.response", {
                            "req_id": req_id,
                            **result,
                        })
                        continue

                    # === VNC Mode: Disable VNC mode ===
                    if topic == "browser.vnc.disable":
                        req_id = str(data.get("req_id") or "")
                        result = await self.disable_vnc_mode()
                        append_bus_event(self.bus_dir, "browser.vnc.disable.response", {
                            "req_id": req_id,
                            **result,
                        })
                        continue

                    # === VNC Mode: Get status ===
                    if topic == "browser.vnc.status":
                        req_id = str(data.get("req_id") or "")
                        result = self.get_vnc_status()
                        append_bus_event(self.bus_dir, "browser.vnc.status.response", {
                            "req_id": req_id,
                            **result,
                        })
                        continue

                    # === VNC Mode: Navigate to login page ===
                    if topic == "browser.vnc.navigate_login":
                        req_id = str(data.get("req_id") or "")
                        provider_id = str(data.get("provider") or "")
                        if provider_id:
                            result = await self.navigate_to_login(provider_id, req_id=req_id or None)
                            append_bus_event(self.bus_dir, "browser.vnc.navigate_login.response", {
                                "req_id": req_id,
                                **result,
                            })
                        continue

                    # === VNC Mode: Focus provider tab (bring to front) ===
                    if topic == "browser.vnc.focus_tab":
                        req_id = str(data.get("req_id") or "")
                        provider_id = str(data.get("provider") or "")
                        if provider_id:
                            result = await self.focus_tab(provider_id, req_id=req_id or None)
                            append_bus_event(self.bus_dir, "browser.vnc.focus_tab.response", {
                                "req_id": req_id,
                                **result,
                            })
                        continue

                    # === VNC Mode: Check if login is complete ===
                    if topic == "browser.vnc.check_login":
                        req_id = str(data.get("req_id") or "")
                        provider_id = str(data.get("provider") or "")
                        if provider_id:
                            result = await self.check_login_complete(provider_id)
                            append_bus_event(self.bus_dir, "browser.vnc.check_login.response", {
                                "req_id": req_id,
                                **result,
                            })
                        continue

                    # === Human acknowledgement for login notifications ===
                    if topic == "inbox.human.acknowledge":
                        ack_type = str(data.get("type") or "")
                        provider_id = str(data.get("provider") or "")
                        if ack_type == "login_retry" and provider_id:
                            # Human wants to retry login - clear notification flag
                            if provider_id in self.human_notified_for_login:
                                del self.human_notified_for_login[provider_id]
                            append_bus_event(self.bus_dir, "inbox.human.acknowledge.response", {
                                "type": ack_type,
                                "provider": provider_id,
                                "status": "cleared",
                                "message": f"Login notification cleared for {provider_id}. Will retry on next health check.",
                            })
                        elif ack_type == "login_dismiss" and provider_id:
                            # Human dismisses - extend cooldown significantly (8 hours)
                            self.human_notified_for_login[provider_id] = time.time() + (8 * 3600)
                            append_bus_event(self.bus_dir, "inbox.human.acknowledge.response", {
                                "type": ack_type,
                                "provider": provider_id,
                                "status": "dismissed",
                                "message": f"Login notification dismissed for {provider_id}. Will not retry for 8 hours.",
                            })
                        continue

                    # === Chat request handling (existing) ===
                    if topic != "browser.chat.request":
                        continue

                    req_id = str(data.get("req_id") or "")
                    provider_id = str(data.get("provider") or "")
                    prompt = str(data.get("prompt") or "")
                    if not req_id or not provider_id or not prompt:
                        continue

                    result = await self.send_message(provider_id, prompt)
                    payload = {
                        "req_id": req_id,
                        "provider": provider_id,
                        "success": bool(result.get("success")),
                        "elapsed_s": result.get("elapsed_s"),
                    }
                    if result.get("success"):
                        full = str(result.get("response") or "")
                        max_chars = int(os.environ.get("PLURIBUS_BROWSER_RESPONSE_MAX_CHARS") or "12000")
                        payload["response_len"] = len(full)
                        payload["response_preview"] = full[:500]
                        payload["response"] = full[:max_chars]
                        payload["response_truncated"] = len(full) > max_chars
                    else:
                        payload["error"] = result.get("error") or "unknown error"
                    append_bus_event(self.bus_dir, "browser.chat.response", payload, actor=provider_to_actor(payload.get("provider")))

                self._save_cursor(pos)

            except asyncio.CancelledError:
                break
            except Exception as e:
                append_bus_event(self.bus_dir, "browser.bus_loop.error", {"error": str(e)[:200]})

    async def health_check(self) -> dict:
        """Check health of all tabs."""
        results = {}

        for provider_id, tab in self.state.tabs.items():
            page = self.pages.get(provider_id)
            if not page:
                tab.status = "closed"
                results[provider_id] = {"status": "closed", "error": "Page not found"}
                continue

            try:
                # Check if page is still responsive.
                # Playwright can transiently throw "Execution context was destroyed" during navigations/refreshes.
                # Treat this as a retryable condition rather than hard failure.
                for attempt in range(3):
                    try:
                        await page.evaluate("() => document.readyState")
                        break
                    except Exception as e:
                        msg = str(e)
                        if "Execution context was destroyed" in msg or "most likely because of a navigation" in msg:
                            try:
                                await page.wait_for_load_state("domcontentloaded", timeout=15_000)
                            except Exception:
                                pass
                            await asyncio.sleep(0.25)
                            continue
                        raise

                # Check session age
                session_start = datetime.fromisoformat(tab.session_start.replace("Z", "+00:00"))
                session_age = datetime.utcnow().replace(tzinfo=session_start.tzinfo) - session_start
                max_hours = WEB_PROVIDERS[provider_id]["max_session_hours"]

                try:
                    tab.current_url = page.url
                except Exception:
                    pass
                try:
                    tab.title = await page.title()
                except Exception:
                    pass

                current_url = (tab.current_url or "").strip()
                current_url_lower = current_url.lower()
                allowed_hosts = _allowed_hosts_for_config(WEB_PROVIDERS.get(provider_id, {}))
                current_host = _url_host(current_url)
                canonical_url = str(WEB_PROVIDERS.get(provider_id, {}).get("url") or "").strip()

                # Guardrail: if the tab drifted to a different site, bring it back to the canonical chat URL.
                if canonical_url and allowed_hosts and ((not current_host) or current_host not in allowed_hosts) and not _looks_like_login_flow(
                    current_url, login_url=WEB_PROVIDERS.get(provider_id, {}).get("login_url")
                ):
                    try:
                        await page.goto(canonical_url, wait_until="domcontentloaded", timeout=30_000)
                        tab.current_url = page.url
                        try:
                            tab.title = await page.title()
                        except Exception:
                            pass
                        current_url = (tab.current_url or "").strip()
                        current_url_lower = current_url.lower()
                        current_host = _url_host(current_url)
                    except Exception:
                        pass

                if session_age > timedelta(hours=max_hours):
                    tab.status = "expired"
                    tab.error = f"Session expired (>{max_hours}h)"
                elif tab.title and "just a moment" in tab.title.lower():
                    if await self._maybe_clear_bot_challenge(page, tab, wait_s=20):
                        tab.status = "ready"
                        tab.error = None
                    else:
                        tab.status = "blocked_bot"
                        tab.error = f"Bot challenge detected at {current_url}"
                elif provider_id == "chatgpt-web" and await self._chatgpt_noauth_present(page):
                    tab.status = "needs_login"
                    tab.error = f"Login required at {current_url}"
                elif provider_id == "gemini-web" and "/welcome" in current_url_lower:
                    try:
                        login_link = await page.query_selector("a[href*='accounts.google.com']")
                    except Exception:
                        login_link = None
                    if login_link:
                        tab.status = "needs_login"
                        tab.error = f"Login required at {current_url}"
                    else:
                        # AUTO-ONBOARDING: Try to click through welcome screen
                        onboarding_clicked = False
                        for btn_selector in [
                            'button:has-text("Get started")',
                            'button:has-text("Build")',
                            'button:has-text("Continue")',
                            'button:has-text("Start")',
                            '[data-testid="get-started"]',
                            'a:has-text("Get started")',
                        ]:
                            try:
                                btn = page.locator(btn_selector).first
                                if await btn.is_visible(timeout=2000):
                                    await btn.click(timeout=5000)
                                    onboarding_clicked = True
                                    await asyncio.sleep(2)
                                    break
                            except Exception:
                                continue

                        # Re-check URL after click attempt
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=10_000)
                            current_url = page.url
                            tab.current_url = current_url
                        except Exception:
                            pass

                        if "/welcome" not in current_url.lower():
                            tab.status = "ready"
                            tab.error = None
                            append_bus_event(self.bus_dir, "browser.vnc.onboarding_complete", {
                                "provider": provider_id,
                                "url": current_url,
                                "context": "health_check"
                            })
                        else:
                            tab.status = "needs_onboarding"
                            tab.error = f"Onboarding required at {current_url}"
                elif _looks_like_login_flow(current_url, login_url=WEB_PROVIDERS.get(provider_id, {}).get("login_url")):
                    tab.status = "needs_login"
                    tab.error = f"Login required at {current_url}"
                elif allowed_hosts and ((not current_host) or current_host not in allowed_hosts):
                    tab.status = "error"
                    tab.error = f"Unexpected URL host ({current_host})"
                else:
                    # Default: ready unless provider-specific checks find a blocker.
                    tab.status = "ready"
                    tab.error = None

                    if provider_id == "claude-web":
                        # Claude can temporarily serve an outage page that has no chat box.
                        try:
                            html = await page.content()
                        except Exception:
                            html = ""
                        if _looks_like_claude_service_disruption(html):
                            tab.status = "error"
                            tab.error = "Claude service disruption (no chat UI available)"

                    if provider_id == "gemini-web" and tab.status == "ready":
                        # AI Studio can require project/prompt selection before the chat UI appears.
                        try:
                            await self._try_gemini_project_picker(page, tab)
                        except Exception:
                            pass

                tab.last_health_check = now_iso()
                results[provider_id] = {"status": tab.status, "age_hours": session_age.total_seconds() / 3600}

            except Exception as e:
                tab.status = "error"
                tab.error = str(e)
                results[provider_id] = {"status": "error", "error": str(e)}

        self.save_state()
        return results

    async def maybe_auto_login(self) -> None:
        """Attempt env-driven auto-login for tabs stuck at needs_login or needs_project.

        IMPORTANT: To prevent email spam loops (e.g., Anthropic magic links),
        we only attempt auto-login ONCE per provider, then emit a human notification
        and wait for human acknowledgement before retrying.
        """
        for provider_id, tab in self.state.tabs.items():
            if not tab:
                continue

            # Handle needs_project separately (no credentials needed, just click-through)
            if tab.status == "needs_project" and provider_id == "gemini-web":
                page = self.pages.get(provider_id)
                if page:
                    try:
                        if await self._try_gemini_project_picker(page, tab):
                            append_bus_event(self.bus_dir, "browser.auto_login.project_selected", {
                                "provider": provider_id,
                                "url": page.url,
                            })
                    except Exception as e:
                        append_bus_event(self.bus_dir, "browser.auto_login.project_error", {
                            "provider": provider_id,
                            "error": str(e)[:200],
                        })
                continue

            if tab.status not in ("needs_login", "needs_onboarding"):
                # Clear notification flag if status changed (login succeeded or provider ready)
                if provider_id in self.human_notified_for_login:
                    del self.human_notified_for_login[provider_id]
                continue

            # Check if we've already notified human about this provider
            last_notified = self.human_notified_for_login.get(provider_id, 0)
            now = time.time()

            if last_notified > 0:
                # Already notified - only re-notify after 1 hour cooldown
                if (now - last_notified) < self.HUMAN_NOTIFICATION_COOLDOWN_S:
                    # Skip auto-login, human has been notified
                    continue
                # Cooldown passed, re-notify but don't spam auto-login
                append_bus_event(self.bus_dir, "inbox.human.action_required", {
                    "type": "login_reminder",
                    "provider": provider_id,
                    "status": tab.status,
                    "message": f"[REMINDER] {provider_id} still needs login. Please check VNC or complete magic link.",
                    "vnc_available": self.state.vnc_mode.enabled,
                    "first_notified_ago_min": int((now - last_notified) / 60),
                })
                self.human_notified_for_login[provider_id] = now
                continue

            # First time seeing needs_login - try auto-login once
            if not self._should_auto_login(provider_id):
                continue

            result = await self._auto_login_provider(provider_id)
            append_bus_event(self.bus_dir, "browser.auto_login.result", {
                "provider": provider_id,
                **{k: v for k, v in result.items() if k not in {"message"}},
            })

            # If login didn't succeed, notify human and stop retrying
            if not result.get("success") and tab.status in ("needs_login", "needs_onboarding", "needs_code"):
                self.human_notified_for_login[provider_id] = now
                append_bus_event(self.bus_dir, "inbox.human.action_required", {
                    "type": "login_needed",
                    "provider": provider_id,
                    "status": tab.status,
                    "message": f"{provider_id} requires manual login. Auto-login attempted once. Check email for magic link or connect via VNC.",
                    "vnc_available": self.state.vnc_mode.enabled,
                    "auto_login_result": result.get("status"),
                })

    async def watchdog_alerts(self) -> None:
        """Emit alerts when all providers are down or blocked."""
        statuses = {pid: tab.status for pid, tab in self.state.tabs.items()}
        if not statuses:
            return
        all_down = all(s in ("needs_login", "needs_onboarding", "needs_code", "needs_project", "blocked_bot", "error", "expired", "closed") for s in statuses.values())
        if all_down:
            append_bus_event(self.bus_dir, "browser.watchdog.alert", {
                "reason": "all_providers_down",
                "statuses": statuses,
            })
            # If VNC is available, auto-enable to speed up manual recovery
            try:
                await self.enable_vnc_mode()
            except Exception:
                pass

    async def send_message(self, provider_id: str, prompt: str) -> dict:
        """Send a message through a web provider tab."""
        config = WEB_PROVIDERS.get(provider_id)
        if not config:
            return {"success": False, "error": f"Unknown provider: {provider_id}"}

        tab = self.state.tabs.get(provider_id)
        if not tab:
            return {"success": False, "error": "Tab not found"}
        allowed_statuses = {"ready", "needs_onboarding", "blocked_bot", "error", "expired", "closed", "needs_project"}
        if tab.status not in allowed_statuses:
            return {"success": False, "error": f"Tab not ready: {tab.status}"}

        page = self.pages.get(provider_id)
        if not page:
            return {"success": False, "error": "Page not found"}

        # Attempt quick recovery from prior error state by reloading to the canonical URL.
        if tab.status in ("error", "expired", "closed"):
            try:
                await page.goto(config["url"], wait_until="domcontentloaded", timeout=30_000)
            except Exception:
                pass
            tab.status = "ready"
            tab.error = None
            self.save_state()

        tab.status = "busy"
        self.save_state()

        start_time = time.time()

        try:
            # If we're on a login page, fail fast with a clear status.
            try:
                tab.current_url = page.url
            except Exception:
                tab.current_url = ""
            try:
                tab.title = await page.title()
            except Exception:
                tab.title = ""
            login_url = str(config.get("login_url") or "").strip()
            if tab.title and "just a moment" in tab.title.lower():
                if await self._maybe_clear_bot_challenge(page, tab, wait_s=20):
                    tab.status = "ready"
                    tab.error = None
                    self.save_state()
                else:
                    tab.status = "blocked_bot"
                    tab.error = f"Bot challenge detected at {tab.current_url}"
                    self.save_state()
                    return {"success": False, "error": tab.error}
            if tab.current_url and (
                ("login" in tab.current_url.lower())
                or ("auth" in tab.current_url.lower())
                or ("accounts" in tab.current_url.lower())
                or (login_url and login_url in tab.current_url)
            ):
                tab.status = "needs_login"
                tab.error = f"Login required at {tab.current_url}"
                self.save_state()
                return {"success": False, "error": tab.error}
            if provider_id == "chatgpt-web":
                if not await self._maybe_dismiss_chatgpt_noauth(page):
                    tab.status = "needs_login"
                    tab.error = f"Login required at {tab.current_url}"
                    self.save_state()
                    return {"success": False, "error": tab.error}

            # Guardrail: keep providers on their canonical host so we don't type prompts into random pages.
            allowed_hosts = _allowed_hosts_for_config(config)
            current_host = _url_host(tab.current_url)
            if allowed_hosts and ((not current_host) or current_host not in allowed_hosts):
                try:
                    await page.goto(config["url"], wait_until="domcontentloaded", timeout=30_000)
                    tab.current_url = page.url
                    try:
                        tab.title = await page.title()
                    except Exception:
                        tab.title = ""
                    current_host = _url_host(tab.current_url)
                except Exception:
                    pass
                # Re-check for login redirects after navigation.
                if _looks_like_login_flow(tab.current_url, login_url=str(config.get("login_url") or "").strip()):
                    tab.status = "needs_login"
                    tab.error = f"Login required at {tab.current_url}"
                    self.save_state()
                    return {"success": False, "error": tab.error}
                if allowed_hosts and ((not current_host) or current_host not in allowed_hosts):
                    tab.status = "error"
                    tab.error = f"Unexpected URL host ({current_host})"
                    self.save_state()
                    return {"success": False, "error": tab.error}

            # Gemini AI Studio can land on /welcome which requires a click-through.
            if provider_id == "gemini-web" and tab.current_url and "/welcome" in tab.current_url:
                # If the welcome screen is unauthenticated, fail fast.
                try:
                    login_link = await page.query_selector("a[href*='accounts.google.com']")
                except Exception:
                    login_link = None
                if login_link:
                    tab.status = "needs_login"
                    tab.error = f"Login required at {tab.current_url}"
                    self.save_state()
                    return {"success": False, "error": tab.error}

                # AUTO-ONBOARDING: Click through welcome screen
                onboarding_clicked = False
                for btn_selector in [
                    'button:has-text("Get started")',
                    'button:has-text("Build")',
                    'button:has-text("Continue")',
                    'button:has-text("Start")',
                    '[data-testid="get-started"]',
                    'a:has-text("Get started")',
                ]:
                    try:
                        btn = page.locator(btn_selector).first
                        if await btn.is_visible(timeout=2000):
                            await btn.click(timeout=5000)
                            onboarding_clicked = True
                            await asyncio.sleep(2)
                            break
                    except Exception:
                        continue

                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=30_000)
                    tab.current_url = page.url
                except Exception:
                    pass

                # Check if onboarding successful
                if "/welcome" not in (tab.current_url or "").lower():
                    tab.status = "ready"
                    tab.error = None
                    append_bus_event(self.bus_dir, "browser.vnc.onboarding_complete", {
                        "provider": provider_id,
                        "url": tab.current_url,
                        "context": "send_message"
                    })
                else:
                    tab.status = "needs_onboarding"
                    tab.error = f"Onboarding required at {tab.current_url}"
                self.save_state()
                if tab.status != "ready":
                    return {"success": False, "error": tab.error or "Onboarding required"}

            if provider_id == "claude-web":
                # Anthropic can return a temporary outage page with no chat input.
                try:
                    html = await page.content()
                except Exception:
                    html = ""
                if _looks_like_claude_service_disruption(html):
                    # Try a single reload before failing (often transient).
                    try:
                        await page.reload(wait_until="domcontentloaded", timeout=30_000)
                        html = await page.content()
                    except Exception:
                        pass
                    if _looks_like_claude_service_disruption(html):
                        tab.status = "error"
                        tab.error = "Claude service disruption (no chat UI available)"
                        self.save_state()
                        return {"success": False, "error": tab.error}

            try:
                await page.wait_for_load_state("domcontentloaded", timeout=15_000)
            except Exception:
                pass

            if provider_id == "gemini-web":
                # AI Studio may require project selection before the chat box appears.
                try:
                    await self._try_gemini_project_picker(page, tab)
                except Exception:
                    pass
                if tab.status == "needs_project":
                    self.save_state()
                    return {"success": False, "error": tab.error or "Project selection required"}

            # Find and fill chat input
            chat_input = await page.wait_for_selector(config["chat_selector"], timeout=20000)
            await chat_input.fill(prompt)

            # Snapshot response list state before sending (avoid matching prior messages).
            responses = page.locator(config["response_selector"])
            try:
                prev_count = await responses.count()
            except Exception:
                prev_count = 0
            prev_last = ""
            if prev_count > 0:
                try:
                    prev_last = (await responses.nth(prev_count - 1).inner_text()) or ""
                except Exception:
                    prev_last = ""

            # Find and click submit
            submit_btn = await page.wait_for_selector(config["submit_selector"], timeout=5000)
            await submit_btn.click()

            # Wait for response (new assistant message or updated last message).
            await asyncio.sleep(2)  # Brief delay for response to start
            response_text = ""
            deadline = time.time() + 60
            while time.time() < deadline:
                await asyncio.sleep(1)
                try:
                    count = await responses.count()
                except Exception:
                    count = 0
                if count <= 0:
                    continue
                idx = count - 1
                try:
                    candidate = await responses.nth(idx).inner_text()
                except Exception:
                    try:
                        candidate = await responses.nth(idx).text_content()
                    except Exception:
                        candidate = ""
                candidate = (candidate or "").strip()
                if not candidate:
                    continue
                if count > prev_count:
                    response_text = candidate
                    break
                if candidate != (prev_last or "").strip():
                    response_text = candidate
                    break

            if not response_text:
                # If ChatGPT is showing the logged-out modal, report needs_login instead of a misleading empty response.
                if provider_id == "chatgpt-web" and await self._chatgpt_noauth_present(page):
                    tab.status = "needs_login"
                    tab.error = f"Login required at {tab.current_url}"
                    self.save_state()
                    return {"success": False, "error": tab.error}
                await self.save_snapshot(page, provider_id, "empty_response", "Empty response extracted")
                raise RuntimeError("Empty response extracted")

            elapsed = time.time() - start_time

            tab.status = "ready"
            tab.last_activity = now_iso()
            tab.chat_count += 1
            self.save_state()

            # Emit transcript to bus
            append_bus_event(self.bus_dir, "browser.chat.completed", {
                "provider": provider_id,
                "prompt": prompt[:200],
                "response_preview": response_text[:500] if response_text else "",
                "elapsed_s": round(elapsed, 2),
                "chat_count": tab.chat_count,
            })

            return {
                "success": True,
                "provider": provider_id,
                "response": response_text,
                "elapsed_s": elapsed,
            }

        except Exception as e:
            tab.status = "error"
            tab.error = str(e)
            # Debug artifacts (local-only): screenshot + minimal page metadata.
            try:
                await self.save_snapshot(page, provider_id, "chat_failure", str(e))
            except Exception:
                pass

            self.save_state()

            append_bus_event(
                self.bus_dir,
                "browser.chat.error",
                {"provider": provider_id, "prompt": prompt[:200], "error": str(e)},
            )

            return {"success": False, "error": str(e)}

    async def save_snapshot(self, page: Any, provider_id: str, tag: str, error_msg: str) -> None:
        """Captures a screenshot and page content for debugging/learning."""
        debug_dir = self.root / ".pluribus" / "browser_data" / "snapshots"
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        base = f"{provider_id}_{ts}_{tag}"

        screenshot_path = debug_dir / f"{base}.png"
        html_path = debug_dir / f"{base}.html"
        meta_path = debug_dir / f"{base}.json"

        try:
            current_url = page.url
        except Exception:
            current_url = "unknown"
        try:
            title = await page.title()
        except Exception:
            title = "unknown"

        screenshot_ok = True
        try:
            await page.screenshot(path=str(screenshot_path), full_page=True)
        except Exception:
            screenshot_ok = False

        html_ok = True
        try:
            html_content = await page.content()
            html_path.write_text(html_content, encoding="utf-8")
        except Exception:
            html_ok = False

        meta = {
            "provider": provider_id,
            "tag": tag,
            "url": current_url,
            "title": title,
            "error": error_msg,
            "screenshot": str(screenshot_path) if screenshot_ok else None,
            "html_dump": str(html_path) if html_ok else None,
            "timestamp": ts,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        append_bus_event(self.bus_dir, "browser.snapshot.captured", {k: v for k, v in meta.items() if k != "error"})
        return {
            "screenshot": meta.get("screenshot"),
            "html": meta.get("html_dump"),
            "url": current_url,
            "title": title,
        }

    async def solve_challenge(self, page: Any, provider_id: str, reason: str) -> dict:
        """
        Attempt to solve a challenge (captcha/2FA) via external command or human-in-loop bus.
        Does not store secrets; returns applied:bool, answer: str|None.
        """
        # Capture snapshot to supply context (for solver + operator).
        snap = await self.save_snapshot(page, provider_id, f"challenge_{reason}", reason)
        html_path = Path(snap.get("html") or "")
        screenshot_path = Path(snap.get("screenshot") or "")

        req_id = str(uuid.uuid4())

        # Start tailing the bus from "now" (avoid scanning the entire events.ndjson).
        pos = 0
        try:
            if self.events_path.exists():
                pos = int(self.events_path.stat().st_size)
        except Exception:
            pos = 0

        append_bus_event(
            self.bus_dir,
            "browser.challenge.request",
            {
                "req_id": req_id,
                "provider": provider_id,
                "reason": reason,
                "snapshot": snap,
            },
        )

        def _extract_json_dict(text: str) -> dict | None:
            s = (text or "").strip()
            if not s:
                return None
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass
            start = s.find("{")
            end = s.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                obj = json.loads(s[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

        def _decode_payload(raw: str) -> tuple[str | None, list[dict] | None, dict | None]:
            """
            Interpret solver/bus payload.
            - Plain text => answer
            - JSON => may include answer/actions
            """
            s = (raw or "").strip()
            if not s:
                return None, None, None
            obj = _extract_json_dict(s)
            if not obj:
                return s, None, None
            answer = obj.get("answer")
            if isinstance(answer, str) and answer.strip():
                answer_s = answer.strip()
            else:
                answer_s = None
            actions = obj.get("actions")
            actions_list: list[dict] | None = None
            if isinstance(actions, list):
                filtered: list[dict] = [a for a in actions if isinstance(a, dict)]
                actions_list = filtered or None
            return answer_s, actions_list, obj

        async def _apply_actions(actions: list[dict]) -> bool:
            """Apply a minimal action DSL in Playwright (safe subset)."""
            if not actions:
                return False

            # Never automate CAPTCHAs / bot checks.
            html_text = ""
            try:
                html_text = html_path.read_text(encoding="utf-8", errors="replace") if html_path.exists() else ""
            except Exception:
                html_text = ""
            t = html_text.lower()
            if any(x in t for x in ("captcha", "recaptcha", "hcaptcha", "cf-challenge", "turnstile")):
                return False

            applied_any = False
            for act in actions:
                typ = str(act.get("type") or act.get("action") or "").strip().lower()
                if typ in ("", "noop"):
                    continue

                # Reject obvious captcha selectors even if the HTML heuristic missed it.
                sel = str(act.get("selector") or "").strip()
                if sel and any(x in sel.lower() for x in ("captcha", "recaptcha", "hcaptcha", "turnstile", "cf-challenge")):
                    continue

                try:
                    if typ == "wait":
                        seconds = float(act.get("seconds") or act.get("s") or act.get("wait_s") or 1)
                        await asyncio.sleep(max(0.0, min(120.0, seconds)))
                        applied_any = True
                        continue

                    if typ == "press":
                        key = str(act.get("key") or "").strip()
                        if key:
                            await page.keyboard.press(key)
                            applied_any = True
                        continue

                    if typ == "goto":
                        url = str(act.get("url") or "").strip()
                        if url:
                            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                            applied_any = True
                        continue

                    if typ == "click":
                        if sel:
                            await page.click(sel, timeout=5000)
                            applied_any = True
                            continue
                        x = act.get("x")
                        y = act.get("y")
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                            await page.mouse.click(int(x), int(y))
                            applied_any = True
                        continue

                    if typ in ("fill", "type"):
                        text = act.get("text")
                        if not isinstance(text, str) or not text:
                            continue
                        if sel:
                            await page.fill(sel, text)
                            applied_any = True
                            continue
                        # Fallback: type at current focus.
                        delay = int(act.get("delay_ms") or 35)
                        await page.keyboard.type(text, delay=max(0, min(250, delay)))
                        applied_any = True
                        continue
                except Exception:
                    continue
            return applied_any

        async def _apply_answer(answer: str) -> bool:
            if not answer:
                return False
            # Try common OTP input selectors first, then fall back to generic inputs.
            selectors = [
                'input[name="totpPin"]',
                'input[autocomplete="one-time-code"]',
                "input[type='tel']",
                "input[type='text']",
                "input[type='number']",
            ]
            filled = False
            for sel in selectors:
                try:
                    await page.fill(sel, answer)
                    filled = True
                    break
                except Exception:
                    continue
            if not filled:
                try:
                    await page.keyboard.type(answer, delay=35)
                    filled = True
                except Exception:
                    filled = False
            try:
                await page.keyboard.press("Enter")
            except Exception:
                pass
            return filled

        answer: str | None = None
        applied = False
        solver_raw: str | None = None
        solver_obj: dict | None = None
        solver_actions: list[dict] | None = None

        # External solver hook (if configured). Prefer structured output (JSON).
        if SOLVER_ENABLED and SOLVER_CMD:
            solver_raw = _run_solver(
                SOLVER_CMD,
                html_path,
                screenshot_path,
                provider_id,
                reason=reason,
                req_id=req_id,
            )
            if solver_raw:
                ans, acts, obj = _decode_payload(solver_raw)
                solver_obj = obj
                solver_actions = acts
                if ans:
                    answer = ans

        if solver_actions and not answer:
            try:
                applied = await _apply_actions(solver_actions)
            except Exception:
                applied = False

        if answer:
            try:
                applied = await _apply_answer(answer)
            except Exception:
                applied = False

        # Human-in-loop via bus response (non-blocking; waits longer for OTP).
        reason_l = (reason or "").lower()
        default_wait_s = 120 if any(k in reason_l for k in ("otp", "2fa", "code")) else 20
        wait_s = int(os.environ.get("PLURIBUS_CHALLENGE_WAIT_S") or str(default_wait_s))
        poll_s = float(os.environ.get("PLURIBUS_CHALLENGE_POLL_S") or "1.0")
        max_read_bytes = int(os.environ.get("PLURIBUS_CHALLENGE_MAX_READ_BYTES") or str(512 * 1024))

        buf = b""
        deadline = time.time() + max(1, wait_s)
        while time.time() < deadline and not answer:
            await asyncio.sleep(max(0.2, min(2.0, poll_s)))
            try:
                if not self.events_path.exists():
                    continue
                # Handle file truncation/rotation defensively.
                try:
                    size = int(self.events_path.stat().st_size)
                except Exception:
                    size = 0
                if size < pos:
                    pos = 0

                with self.events_path.open("rb") as f:
                    f.seek(pos)
                    chunk = f.read(max_read_bytes)
                    pos = f.tell()
            except Exception:
                continue
            if not chunk:
                continue

            buf += chunk
            parts = buf.split(b"\n")
            buf = parts.pop()  # remainder (possibly partial line)
            for raw in parts:
                if not raw:
                    continue
                try:
                    ev = json.loads(raw.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if ev.get("topic") != "browser.challenge.response":
                    continue
                data = ev.get("data") or {}
                if not isinstance(data, dict) or data.get("req_id") != req_id:
                    continue
                candidate = data.get("answer")
                if isinstance(candidate, str):
                    cand_s = candidate.strip()
                else:
                    cand_s = ""
                if not cand_s:
                    continue
                answer = cand_s
                break

            if answer:
                try:
                    applied = await _apply_answer(answer)
                except Exception:
                    applied = False
                break

        append_bus_event(
            self.bus_dir,
            "browser.challenge.result",
            {
                "req_id": req_id,
                "provider": provider_id,
                "reason": reason,
                "applied": bool(applied),
                "had_answer": bool(answer),
                "solver_used": bool(solver_raw),
                "solver_had_actions": bool(solver_actions),
                "solver_had_json": bool(solver_obj),
            },
        )
        return {"applied": bool(applied), "answer": answer, "req_id": req_id}

    async def cleanup_old_snapshots(self) -> None:
        """Removes snapshots older than 24 hours."""
        debug_dir = self.root / ".pluribus" / "browser_data" / "snapshots"
        if not debug_dir.exists():
            return

        now = time.time()
        for f in debug_dir.iterdir():
            if not f.is_file():
                continue
            age_seconds = now - f.stat().st_mtime
            if age_seconds > 24 * 3600:  # 24 hours
                try:
                    f.unlink()
                    append_bus_event(self.bus_dir, "browser.snapshot.cleaned", {"path": str(f)})
                except Exception:
                    pass

    async def stop(self) -> None:
        """Stop the daemon."""
        self._running = False

        stopped_pid = int(self.state.pid or 0)
        for btype, context in list(self.contexts.items()):
            try:
                await context.close()
            except Exception:
                pass
        self.contexts = {}
        try:
            if getattr(self, "playwright", None):
                await self.playwright.stop()
        except Exception:
            pass

        self.state.running = False
        self.state.pid = 0
        self.state.browser_pids = {}
        self.save_state()

        append_bus_event(self.bus_dir, "browser.daemon.stopped", {"pid": stopped_pid})

        print("Browser daemon stopped")

    def get_status(self) -> dict:
        """Get current daemon status."""
        self.load_state()
        result = {
            "running": self.state.running,
            "pid": self.state.pid,
            "started_at": self.state.started_at,
            "tabs": {k: asdict(v) for k, v in self.state.tabs.items()},
            "vnc_mode": asdict(self.state.vnc_mode) if self.state.vnc_mode else {},
        }

        # Add live VNC detection info
        display = detect_display() or self.state.vnc_mode.display
        vnc_running = is_vnc_server_running()
        result["vnc_available"] = {
            "display": display,
            "vnc_detected": vnc_running,
            "connection_info": get_vnc_connection_info() if display else None,
        }

        # Add vnc_info for VNCAuthPanel (with connection details for iframe)
        if self.state.vnc_mode.enabled or vnc_running:
            import socket
            hostname = socket.gethostname()
            result["vnc_info"] = {
                "display": display or ":1",
                "vnc_port": 5901,
                "hostname": hostname,
                "connection_string": f"{hostname}:5901",
                "instructions": "Connect via VNC client or use the embedded noVNC viewer",
            }

        return result


# CLI Interface
async def cmd_start(args) -> int:
    root = Path(args.root)
    bus_dir = Path(args.bus_dir)

    daemon = BrowserSessionDaemon(root, bus_dir)

    # Handle signals
    def signal_handler(sig, frame):
        print("\nShutting down...")
        daemon._running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await daemon.start()
    return 0


async def cmd_status(args) -> int:
    root = Path(args.root)
    bus_dir = Path(args.bus_dir)

    daemon = BrowserSessionDaemon(root, bus_dir)
    status = daemon.get_status()

    # JSON output mode for API integrations
    if getattr(args, 'json', False):
        print(json.dumps(status))
        return 0

    print(f"\nBrowser Session Daemon Status")
    print(f"{'='*40}")
    print(f"Running: {status['running']}")
    print(f"PID: {status['pid']}")
    print(f"Started: {status['started_at']}")

    # VNC Mode status
    vnc_mode = status.get("vnc_mode", {})
    vnc_available = status.get("vnc_available", {})
    print(f"\nVNC Mode:")
    print(f"  Enabled: {vnc_mode.get('enabled', False)}")
    print(f"  Display: {vnc_available.get('display') or 'Not set'}")
    print(f"  VNC Server: {'Detected' if vnc_available.get('vnc_detected') else 'Not detected'}")
    if vnc_available.get("connection_info"):
        conn = vnc_available["connection_info"]
        print(f"  Connect: {conn.get('connection_string', 'N/A')}")
    if vnc_mode.get("login_providers_pending"):
        print(f"  Pending Logins: {', '.join(vnc_mode['login_providers_pending'])}")
    if vnc_mode.get("login_providers_completed"):
        print(f"  Completed Logins: {', '.join(vnc_mode['login_providers_completed'])}")

    print(f"\nTabs:")

    for provider_id, tab in status.get("tabs", {}).items():
        indicator = "+" if tab["status"] == "ready" else "o"
        print(f"  {indicator} {provider_id}: {tab['status']}")
        if tab.get("error"):
            print(f"      Error: {tab['error']}")
        print(f"      Chats: {tab.get('chat_count', 0)}")

    return 0


async def cmd_stop(args) -> int:
    root = Path(args.root)
    bus_dir = Path(args.bus_dir)

    daemon = BrowserSessionDaemon(root, bus_dir)
    daemon.load_state()

    pid = int(daemon.state.pid or 0)
    if pid and daemon.state.running and _is_browser_session_daemon_pid(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to daemon (PID {pid})")
            return 0
        except OSError as e:
            # ESRCH means the process is already gone; treat as stopped and clear below.
            if getattr(e, "errno", None) not in (None, 3):
                print(f"Could not stop daemon: {e}")

    # Either not running, stale state, or PID-reused: clear state without sending signals.
    daemon.state.running = False
    daemon.state.pid = 0
    daemon.state.browser_pids = {}
    for tab in daemon.state.tabs.values():
        if tab.status != "closed":
            tab.status = "closed"
            tab.error = "Daemon not running (stale state)"
    try:
        daemon.save_state()
    except Exception:
        pass

    print("Daemon not running")
    return 0


async def cmd_infer(args) -> int:
    root = Path(args.root)
    bus_dir = Path(args.bus_dir)

    daemon = BrowserSessionDaemon(root, bus_dir)
    daemon.load_state()

    if not daemon.state.running:
        print("Daemon not running. Start with: browser_session_daemon.py start")
        return 1

    req_id = str(uuid.uuid4())
    append_bus_event(bus_dir, "browser.chat.request", {
        "req_id": req_id,
        "provider": args.provider,
        "prompt": args.prompt,
        "timeout_s": 60,
    })

    events_path = bus_dir / "events.ndjson"
    deadline = time.time() + 90
    pos = events_path.stat().st_size if events_path.exists() else 0
    while time.time() < deadline:
        if not events_path.exists():
            await asyncio.sleep(0.2)
            continue
        with events_path.open("rb") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        if chunk:
            for raw in chunk.splitlines():
                try:
                    ev = json.loads(raw.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if (ev.get("topic") or "") != "browser.chat.response":
                    continue
                data = ev.get("data") or {}
                if not isinstance(data, dict):
                    continue
                if str(data.get("req_id") or "") != req_id:
                    continue
                if data.get("success"):
                    print((data.get("response") or data.get("response_preview") or "").strip())
                    return 0
                print(f"error: {data.get('error') or 'unknown'}")
                return 1
        await asyncio.sleep(0.2)

    print("error: timeout waiting for response")
    return 1


async def cmd_inject_auth(args) -> int:
    """Inject CLI OAuth credentials into browser session via bus request."""
    bus_dir = Path(args.bus_dir)

    # Load daemon state to check if running
    state_path = Path(args.root) / ".pluribus" / "browser_daemon.json"
    if state_path.exists():
        data = json.loads(state_path.read_text())
        if not data.get("running"):
            print("Daemon not running. Start with: browser_session_daemon.py start")
            return 1
    else:
        print("Daemon not running. Start with: browser_session_daemon.py start")
        return 1

    req_id = str(uuid.uuid4())
    provider = args.provider

    print(f"Injecting CLI auth for {provider}...")
    print(f"  Request ID: {req_id}")

    # Emit bus event
    append_bus_event(bus_dir, "browser.inject_auth", {
        "req_id": req_id,
        "provider": provider,
    })

    # Wait for response
    events_path = bus_dir / "events.ndjson"
    deadline = time.time() + 30
    pos = events_path.stat().st_size if events_path.exists() else 0

    while time.time() < deadline:
        if not events_path.exists():
            await asyncio.sleep(0.2)
            continue
        with events_path.open("rb") as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        if chunk:
            for raw in chunk.splitlines():
                try:
                    ev = json.loads(raw.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                if (ev.get("topic") or "") != "browser.inject_auth.response":
                    continue
                data = ev.get("data") or {}
                if not isinstance(data, dict):
                    continue
                if str(data.get("req_id") or "") != req_id:
                    continue

                success = data.get("success", False)
                message = data.get("message", "")
                method = data.get("method", "")

                if success:
                    print(f"  ✓ SUCCESS: {message}")
                    print(f"  Method: {method}")
                    return 0
                else:
                    print(f"  ✗ FAILED: {message}")
                    print(f"  Method attempted: {method}")
                    return 1
        await asyncio.sleep(0.2)

    print("error: timeout waiting for inject_auth response")
    return 1


async def cmd_vnc_mode(args) -> int:
    """Control VNC mode for manual OAuth login."""
    root = Path(args.root)
    bus_dir = Path(args.bus_dir)

    daemon = BrowserSessionDaemon(root, bus_dir)
    daemon.load_state()

    action = args.action

    if action == "status":
        # Show VNC mode status (doesn't require daemon to be running)
        display = detect_display()
        vnc_running = is_vnc_server_running()

        print(f"\nVNC Mode Status")
        print(f"{'='*40}")
        print(f"Environment:")
        print(f"  PLURIBUS_BROWSER_VNC: {os.environ.get('PLURIBUS_BROWSER_VNC', 'not set')}")
        print(f"  DISPLAY: {display or 'not set'}")
        print(f"  VNC Server: {'Detected' if vnc_running else 'Not detected'}")

        if display:
            vnc_info = get_vnc_connection_info()
            print(f"\nConnection Info:")
            print(f"  Hostname: {vnc_info['hostname']}")
            print(f"  Port: {vnc_info['vnc_port']}")
            print(f"  Connect: {vnc_info['connection_string']}")

        print(f"\nDaemon VNC State:")
        vnc_state = daemon.state.vnc_mode
        print(f"  Enabled: {vnc_state.enabled}")
        print(f"  Started at: {vnc_state.started_at or 'N/A'}")
        print(f"  Pending logins: {', '.join(vnc_state.login_providers_pending) or 'None'}")
        print(f"  Completed logins: {', '.join(vnc_state.login_providers_completed) or 'None'}")

        # Show providers needing login
        needs_action = []
        for provider_id, tab in daemon.state.tabs.items():
            if tab.status in ("needs_login", "needs_onboarding", "needs_code", "blocked_bot"):
                needs_action.append(f"{provider_id} ({tab.status})")
        if needs_action:
            print(f"\nProviders needing login:")
            for item in needs_action:
                print(f"  - {item}")
        else:
            print(f"\nAll providers ready (no login needed)")

        return 0

    elif action == "start":
        if not daemon.state.running:
            print("Daemon not running. Start with: browser_session_daemon.py start")
            print("Tip: Set PLURIBUS_BROWSER_VNC=1 before starting the daemon")
            return 1

        # Send enable request via bus
        req_id = str(uuid.uuid4())
        append_bus_event(bus_dir, "browser.vnc.enable", {"req_id": req_id})

        # Wait for response
        events_path = bus_dir / "events.ndjson"
        deadline = time.time() + 30
        pos = events_path.stat().st_size if events_path.exists() else 0

        while time.time() < deadline:
            if not events_path.exists():
                await asyncio.sleep(0.2)
                continue
            with events_path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                for raw in chunk.splitlines():
                    try:
                        ev = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if (ev.get("topic") or "") != "browser.vnc.enable.response":
                        continue
                    data = ev.get("data") or {}
                    if str(data.get("req_id") or "") != req_id:
                        continue

                    if data.get("success"):
                        print(f"\nVNC Mode Enabled")
                        print(f"{'='*40}")
                        print(f"Display: {data.get('display')}")
                        conn = data.get("connection_info", {})
                        if conn:
                            print(f"Connect via VNC: {conn.get('connection_string')}")
                        pending = data.get("providers_pending", [])
                        if pending:
                            print(f"\nProviders needing login:")
                            for p in pending:
                                print(f"  - {p}")
                            print(f"\nInstructions:")
                            print(f"  1. Connect to VNC at {conn.get('connection_string', 'localhost:5901')}")
                            print(f"  2. Complete OAuth login for each provider")
                            print(f"  3. Run: browser_session_daemon.py vnc-mode check <provider>")
                            print(f"  4. When done: browser_session_daemon.py vnc-mode stop")
                        else:
                            print(f"\nAll providers already logged in!")
                        return 0
                    else:
                        print(f"Error: {data.get('error', 'Unknown error')}")
                        if data.get("hint"):
                            print(f"Hint: {data['hint']}")
                        return 1
            await asyncio.sleep(0.2)

        print("error: timeout waiting for VNC enable response")
        return 1

    elif action == "stop":
        if not daemon.state.running:
            print("Daemon not running")
            return 1

        # Send disable request via bus
        req_id = str(uuid.uuid4())
        append_bus_event(bus_dir, "browser.vnc.disable", {"req_id": req_id})

        # Wait for response
        events_path = bus_dir / "events.ndjson"
        deadline = time.time() + 30
        pos = events_path.stat().st_size if events_path.exists() else 0

        while time.time() < deadline:
            if not events_path.exists():
                await asyncio.sleep(0.2)
                continue
            with events_path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                for raw in chunk.splitlines():
                    try:
                        ev = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if (ev.get("topic") or "") != "browser.vnc.disable.response":
                        continue
                    data = ev.get("data") or {}
                    if str(data.get("req_id") or "") != req_id:
                        continue

                    print(f"\nVNC Mode Disabled")
                    print(f"{'='*40}")
                    completed = data.get("providers_completed", [])
                    pending = data.get("providers_still_pending", [])
                    if completed:
                        print(f"Successfully logged in: {', '.join(completed)}")
                    if pending:
                        print(f"Still need login: {', '.join(pending)}")
                    return 0
            await asyncio.sleep(0.2)

        print("error: timeout waiting for VNC disable response")
        return 1

    elif action == "check":
        if not daemon.state.running:
            print("Daemon not running")
            return 1

        provider = getattr(args, "provider", None)
        if not provider:
            print("error: provider required for check action")
            print("Usage: browser_session_daemon.py vnc-mode check <provider>")
            return 1

        # Send check login request via bus
        req_id = str(uuid.uuid4())
        append_bus_event(bus_dir, "browser.vnc.check_login", {
            "req_id": req_id,
            "provider": provider,
        })

        # Wait for response
        events_path = bus_dir / "events.ndjson"
        deadline = time.time() + 30
        pos = events_path.stat().st_size if events_path.exists() else 0

        while time.time() < deadline:
            if not events_path.exists():
                await asyncio.sleep(0.2)
                continue
            with events_path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                for raw in chunk.splitlines():
                    try:
                        ev = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if (ev.get("topic") or "") != "browser.vnc.check_login.response":
                        continue
                    data = ev.get("data") or {}
                    if str(data.get("req_id") or "") != req_id:
                        continue

                    if data.get("login_complete"):
                        print(f"[OK] {provider}: Login complete!")
                        print(f"  Status: {data.get('status', 'ready')}")
                        return 0
                    else:
                        print(f"[PENDING] {provider}: Login not complete")
                        print(f"  Reason: {data.get('reason', 'unknown')}")
                        print(f"  URL: {data.get('current_url', 'N/A')}")
                        print(f"  Complete the login via VNC and run this check again.")
                        return 1
            await asyncio.sleep(0.2)

        print("error: timeout waiting for check_login response")
        return 1

    elif action == "focus":
        if not daemon.state.running:
            print("Daemon not running")
            return 1

        provider = getattr(args, "provider", None)
        if not provider:
            print("error: provider required for focus action")
            print("Usage: browser_session_daemon.py vnc-mode focus <provider>")
            return 1

        req_id = str(uuid.uuid4())
        append_bus_event(
            bus_dir,
            "browser.vnc.focus_tab",
            {
                "req_id": req_id,
                "provider": provider,
            },
        )

        events_path = bus_dir / "events.ndjson"
        deadline = time.time() + 30
        pos = events_path.stat().st_size if events_path.exists() else 0

        while time.time() < deadline:
            if not events_path.exists():
                await asyncio.sleep(0.2)
                continue
            with events_path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                for raw in chunk.splitlines():
                    try:
                        ev = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if (ev.get("topic") or "") != "browser.vnc.focus_tab.response":
                        continue
                    data = ev.get("data") or {}
                    if str(data.get("req_id") or "") != req_id:
                        continue

                    if data.get("success"):
                        print(f"Focused {provider} tab")
                        print(f"  URL: {data.get('current_url', 'N/A')}")
                        print(f"  Title: {data.get('title', 'N/A')}")
                        return 0

                    print(f"Error: {data.get('error', 'Unknown error')}")
                    return 1
            await asyncio.sleep(0.2)

        print("error: timeout waiting for focus response")
        return 1

    elif action == "navigate":
        if not daemon.state.running:
            print("Daemon not running")
            return 1

        provider = getattr(args, "provider", None)
        if not provider:
            print("error: provider required for navigate action")
            print("Usage: browser_session_daemon.py vnc-mode navigate <provider>")
            return 1

        # Send navigate request via bus
        req_id = str(uuid.uuid4())
        append_bus_event(bus_dir, "browser.vnc.navigate_login", {
            "req_id": req_id,
            "provider": provider,
        })

        # Wait for response
        events_path = bus_dir / "events.ndjson"
        deadline = time.time() + 30
        pos = events_path.stat().st_size if events_path.exists() else 0

        while time.time() < deadline:
            if not events_path.exists():
                await asyncio.sleep(0.2)
                continue
            with events_path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                for raw in chunk.splitlines():
                    try:
                        ev = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if (ev.get("topic") or "") != "browser.vnc.navigate_login.response":
                        continue
                    data = ev.get("data") or {}
                    if str(data.get("req_id") or "") != req_id:
                        continue

                    if data.get("success"):
                        print(f"Navigated {provider} to login page")
                        print(f"  URL: {data.get('current_url', 'N/A')}")
                        print(f"  Title: {data.get('title', 'N/A')}")
                        print(f"\nComplete login via VNC, then run:")
                        print(f"  browser_session_daemon.py vnc-mode check {provider}")
                        return 0
                    else:
                        print(f"Error: {data.get('error', 'Unknown error')}")
                        return 1
            await asyncio.sleep(0.2)

        print("error: timeout waiting for navigate response")
        return 1

    else:
        print(f"Unknown action: {action}")
        print("Valid actions: start, stop, status, check, focus, navigate")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Browser Session Daemon")
    parser.add_argument("--root", default="/pluribus", help="Pluribus root directory")
    parser.add_argument("--bus-dir", default="/pluribus/.pluribus/bus", help="Bus directory")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the daemon")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check daemon status")
    status_parser.add_argument("--json", action="store_true", help="Output status as JSON")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the daemon")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Test inference")
    infer_parser.add_argument("provider", help="Provider ID (gemini-web, claude-web, chatgpt-web)")
    infer_parser.add_argument("prompt", help="Prompt to send")

    # Inject auth command - bridges CLI OAuth to browser sessions
    inject_parser = subparsers.add_parser("inject-auth", help="Inject CLI OAuth credentials into browser session")
    inject_parser.add_argument("provider", help="Provider ID (claude-web, gemini-web, chatgpt-web)")

    # VNC mode command - control VNC mode for manual OAuth login
    vnc_parser = subparsers.add_parser("vnc-mode", help="Control VNC mode for manual OAuth login")
    vnc_parser.add_argument(
        "action",
        choices=["start", "stop", "status", "check", "focus", "navigate"],
        help="Action: start (enable), stop (disable), status, check <provider>, focus <provider>, navigate <provider>",
    )
    vnc_parser.add_argument("provider", nargs="?", default=None,
                           help="Provider ID for check/navigate actions (gemini-web, claude-web, chatgpt-web)")

    args = parser.parse_args()

    if args.command == "start":
        return asyncio.run(cmd_start(args))
    elif args.command == "status":
        return asyncio.run(cmd_status(args))
    elif args.command == "stop":
        return asyncio.run(cmd_stop(args))
    elif args.command == "infer":
        return asyncio.run(cmd_infer(args))
    elif args.command == "inject-auth":
        return asyncio.run(cmd_inject_auth(args))
    elif args.command == "vnc-mode":
        return asyncio.run(cmd_vnc_mode(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
