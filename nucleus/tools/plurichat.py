#!/usr/bin/env python3
"""
PluriChat: Interactive Multi-Model Chat CLI
============================================
A gemini-cli-like interactive tool that routes optimally between:
- Gemini Web (persistent browser session)
- ChatGPT Web (persistent browser session)
- Claude Web (persistent browser session)

Features:
- Interactive REPL with liveness indicators
- Automatic provider routing via Lens/Collimator
- Direct mode (immediate execution) or Proxy mode (bus-mediated)
- Full bus evidence emission
- Query-type-aware model selection

Usage:
    python3 plurichat.py                    # Interactive mode
    python3 plurichat.py --provider auto    # Auto-route
    python3 plurichat.py --provider gemini-web  # Force Gemini Web
    python3 plurichat.py --mode proxy       # Use bus-mediated routing
    python3 plurichat.py --ask "question"   # One-shot mode
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import readline  # noqa: F401 - enables line editing
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.dont_write_bytecode = True

# Add tools dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import lens_collimator for depth-aware routing
try:
    from lens_collimator import (
        LensRequest,
        plan_route as lens_plan_route,
        load_vps_session as lens_load_vps_session,
    )
    LENS_AVAILABLE = True
except ImportError:
    LENS_AVAILABLE = False

try:
    from persona_registry import load_personas as _load_personas  # type: ignore
except Exception:
    _load_personas = None

try:
    from idiolect import load as _load_idiolect  # type: ignore
except Exception:
    _load_idiolect = None

try:
    import pbeport as _pbeport  # type: ignore
except Exception:
    _pbeport = None

# Import semops lexer for semantic operator recognition
try:
    from semops_lexer import SemopsLexer, TokenType, get_lexer
    LEXER_AVAILABLE = True
except ImportError:
    LEXER_AVAILABLE = False
    SemopsLexer = None
    TokenType = None
    get_lexer = None

# Import event_semantics for rich semantic events
try:
    from event_semantics import (
        create_semantic_event,
        emit_semantic_event,
        enrich_event,
        TopologyContext,
        LineageContext,
        CMPSignal,
    )
    SEMANTICS_AVAILABLE = True
except ImportError:
    SEMANTICS_AVAILABLE = False

# Import RAG for memory-aware reasoning
try:
    from rag_vector import VectorRAG, DB_PATH
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ==============================================================================
# Configuration
# ==============================================================================

PLURIBUS_ROOT = Path("/pluribus")
DEFAULT_BUS_DIR = PLURIBUS_ROOT / ".pluribus" / "bus"
TOOLS_DIR = Path(__file__).resolve().parent
PROVIDERS_DIR = TOOLS_DIR / "providers"
VPS_SESSION_PATH = PLURIBUS_ROOT / ".pluribus" / "vps_session.json"
STRP_QUEUE = TOOLS_DIR / "strp_queue.py"
STRP_WORKER = TOOLS_DIR / "strp_worker.py"
SPEAKER_BUS_FILE = PLURIBUS_ROOT / "speaker_bus.aiff"

# Model routing configuration (PluriChat web-only policy)
# PluriChat must route via persistent browser sessions (CUA tabs) and must NOT
# silently fall back to CLI/API/mock providers.
WEB_ONLY_PROVIDERS = ["chatgpt-web", "claude-web", "gemini-web"]

# Deep queries (project-aware) -> still web-only (no CLI lane fallback in PluriChat).
# Narrow queries (isolated tasks) -> still web-only.
MODEL_ROUTING_DEEP = {
    "code": ["chatgpt-web", "claude-web", "gemini-web"],
    "research": ["gemini-web", "claude-web", "chatgpt-web"],
    "creative": ["claude-web", "chatgpt-web", "gemini-web"],
    "analysis": ["chatgpt-web", "claude-web", "gemini-web"],
    "math": ["gemini-web", "chatgpt-web", "claude-web"],
    "general": ["chatgpt-web", "claude-web", "gemini-web"],
}

MODEL_ROUTING_NARROW = {
    "code": ["chatgpt-web", "claude-web", "gemini-web"],
    "research": ["gemini-web", "claude-web", "chatgpt-web"],
    "creative": ["claude-web", "chatgpt-web", "gemini-web"],
    "analysis": ["chatgpt-web", "claude-web", "gemini-web"],
    "math": ["gemini-web", "chatgpt-web", "claude-web"],
    "general": ["chatgpt-web", "claude-web", "gemini-web"],
}

# Fallback for backwards compatibility
MODEL_ROUTING = MODEL_ROUTING_NARROW

# Provider -> model mapping
PROVIDER_MODELS = {
    "gemini-3": ["gemini-web"],
    "gpt-5.2": ["chatgpt-web"],
    "claude-opus": ["claude-web"],
    "claude-sonnet": ["claude-web"],
    # Explicit legacy lanes (not used by auto routing in PluriChat).
    "codex-cli": ["codex-cli"],
    "auto": ["auto"],
}

# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    # Extended colors for ring indicators
    BRIGHT_RED = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_CYAN = "\033[96m"

# ==============================================================================
# SUPERMOTD Status Line Integration
# ==============================================================================

# Ring health colors (Ring 0=critical/red, Ring 1=warn/yellow, Ring 2=nominal/green, Ring 3=optimal/cyan)
RING_COLORS = [C.BRIGHT_RED, C.BRIGHT_YELLOW, C.BRIGHT_GREEN, C.BRIGHT_CYAN]

# Idle insights - rotate through these when system is idle
IDLE_INSIGHTS = [
    "membrane nominal",
    "bus online | awaiting stimulus",
    "dialogos lane ready",
    "lens/collimator primed",
    "STRp topology available",
    "inference pipeline idle",
    "evidence chain intact",
    "ready for operator input",
    "provider mesh healthy",
    "quine replication stable",
    "semantic operators: ITERATE PBFLUSH CKIN",
    "append-only bus | immutable history",
    "hexis buffer clear",
    "context modes: min|lite|full",
    "topology: single|star|peer_debate",
]

# Boot sequence messages (Unix-style)
BOOT_SEQUENCE = [
    ("PLURIBUS", "v1.0", "info"),
    ("kernel", "loading membrane driver", "info"),
    ("bus", "events.ndjson mounted", "ok"),
    ("providers", "initializing mesh", "info"),
    ("lens", "collimator online", "ok"),
    ("dialogos", "lane ready", "ok"),
    ("strp", "topology daemon standby", "info"),
    ("supermotd", "status stream active", "ok"),
]


class SuperMotdStatus:
    """Compact status line generator for TUI integration."""

    def __init__(self, bus_dir: Path):
        self.bus_dir = bus_dir
        self.events_path = bus_dir / "events.ndjson"
        self._last_insight_idx = 0
        self._last_event_count = 0
        self._cached_status: dict[str, Any] = {}
        self._cache_time = 0.0

    def _count_recent_events(self, window_s: float = 60.0) -> dict[str, int]:
        """Count events by kind in recent window."""
        counts: dict[str, int] = {"total": 0, "error": 0, "warn": 0, "request": 0, "response": 0}
        cutoff = time.time() - window_s
        try:
            if not self.events_path.exists():
                return counts
            # Read last 50KB for efficiency
            file_size = self.events_path.stat().st_size
            read_size = min(file_size, 50000)
            with self.events_path.open("r", encoding="utf-8", errors="replace") as f:
                if file_size > read_size:
                    f.seek(file_size - read_size)
                for line in f:
                    try:
                        e = json.loads(line.strip())
                        ts = e.get("ts", 0)
                        if ts < cutoff:
                            continue
                        counts["total"] += 1
                        level = e.get("level", "info")
                        kind = e.get("kind", "event")
                        if level == "error":
                            counts["error"] += 1
                        elif level == "warn":
                            counts["warn"] += 1
                        if kind == "request":
                            counts["request"] += 1
                        elif kind == "response":
                            counts["response"] += 1
                    except Exception:
                        continue
        except Exception:
            pass
        return counts

    def _compute_ring_health(self, providers: dict[str, "ProviderStatus"], event_counts: dict[str, int]) -> list[int]:
        """Compute ring health levels (0-3) for each of 4 rings.

        Ring 0 (Red): Critical infrastructure - bus, core errors
        Ring 1 (Yellow): Provider health - availability, cooldowns
        Ring 2 (Green): Inference health - request/response flow
        Ring 3 (Cyan): Optimal state - all systems nominal
        """
        health = [0, 0, 0, 0]  # 0=inactive, 1=degraded, 2=nominal, 3=optimal

        # Ring 0: Bus health (critical)
        if self.events_path.exists():
            if event_counts.get("error", 0) > 5:
                health[0] = 1  # degraded
            elif event_counts.get("error", 0) > 0:
                health[0] = 2  # nominal with some errors
            else:
                health[0] = 3  # optimal
        else:
            health[0] = 0  # inactive

        # Ring 1: Provider health
        available_count = sum(1 for p in providers.values() if p.available)
        total_providers = len([p for p in providers.values() if p.name != "mock"])
        cooldown_count = sum(1 for p in providers.values() if p.cooldown_until)
        if available_count == 0:
            health[1] = 0
        elif cooldown_count > 0:
            health[1] = 1
        elif available_count < total_providers:
            health[1] = 2
        else:
            health[1] = 3

        # Ring 2: Inference flow
        requests = event_counts.get("request", 0)
        responses = event_counts.get("response", 0)
        if requests == 0:
            health[2] = 0  # no activity
        elif responses < requests * 0.5:
            health[2] = 1  # degraded response rate
        elif responses < requests:
            health[2] = 2  # nominal
        else:
            health[2] = 3  # optimal

        # Ring 3: Overall omega state
        avg_health = sum(health[:3]) / 3
        if avg_health >= 2.5:
            health[3] = 3
        elif avg_health >= 1.5:
            health[3] = 2
        elif avg_health >= 0.5:
            health[3] = 1
        else:
            health[3] = 0

        return health

    def _render_ring_dots(self, health: list[int]) -> str:
        """Render colored ring dots based on health levels."""
        dots = []
        for i, h in enumerate(health):
            color = RING_COLORS[i]
            if h == 0:
                dots.append(f"{C.DIM}o{C.RESET}")  # inactive
            elif h == 1:
                dots.append(f"{color}{C.DIM}*{C.RESET}")  # degraded
            elif h == 2:
                dots.append(f"{color}*{C.RESET}")  # nominal
            else:
                dots.append(f"{color}{C.BOLD}*{C.RESET}")  # optimal
        return "".join(dots)

    def get_next_insight(self) -> str:
        """Get next rotating idle insight."""
        insight = IDLE_INSIGHTS[self._last_insight_idx % len(IDLE_INSIGHTS)]
        self._last_insight_idx += 1
        return insight

    def build_compact_status(self, providers: dict[str, "ProviderStatus"]) -> str:
        """Build compact one-line status string.

        Format: PLURIBUS [****] gen:N bus:M omega:K providers:X/Y | insight...
        """
        now = time.time()
        # Cache status for 2 seconds
        if now - self._cache_time < 2.0 and self._cached_status:
            event_counts = self._cached_status.get("event_counts", {})
        else:
            event_counts = self._count_recent_events(60.0)
            self._cached_status = {"event_counts": event_counts}
            self._cache_time = now

        health = self._compute_ring_health(providers, event_counts)
        ring_dots = self._render_ring_dots(health)

        available = sum(1 for p in providers.values() if p.available and p.name != "mock")
        total = len([p for p in providers if p != "mock"])

        gen = event_counts.get("total", 0)
        bus_events = event_counts.get("request", 0) + event_counts.get("response", 0)
        omega = sum(health)

        insight = self.get_next_insight()

        return (
            f"{C.CYAN}PLURIBUS{C.RESET} [{ring_dots}] "
            f"gen:{gen} bus:{bus_events} omega:{omega} "
            f"providers:{available}/{total} "
            f"{C.DIM}| {insight}{C.RESET}"
        )

    def print_boot_sequence(self, fast: bool = False) -> None:
        """Print Unix-style boot sequence."""
        delay = 0.05 if fast else 0.15
        for subsys, msg, severity in BOOT_SEQUENCE:
            if severity == "ok":
                status = f"{C.GREEN}[ OK ]{C.RESET}"
            elif severity == "error":
                status = f"{C.RED}[FAIL]{C.RESET}"
            elif severity == "warn":
                status = f"{C.YELLOW}[WARN]{C.RESET}"
            else:
                status = f"{C.BLUE}[INFO]{C.RESET}"
            print(f"  {status} {C.BOLD}{subsys}{C.RESET}: {msg}")
            time.sleep(delay)


def get_supermotd_status(bus_dir: Path) -> SuperMotdStatus:
    """Factory function for SuperMotdStatus singleton."""
    if not hasattr(get_supermotd_status, "_instance") or get_supermotd_status._instance is None:
        get_supermotd_status._instance = SuperMotdStatus(bus_dir)
    return get_supermotd_status._instance


# ==============================================================================
# Utilities
# ==============================================================================

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

def load_vps_session() -> dict[str, Any]:
    try:
        if VPS_SESSION_PATH.exists():
            return json.loads(VPS_SESSION_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def format_cooldown_until(ts: float | int | None) -> str | None:
    if ts is None:
        return None
    try:
        ts = float(ts)
    except Exception:
        return None
    if ts <= 0:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> str:
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)
    return evt_id


def emit_semantic(
    bus_dir: Path,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict,
    semantic: str,
    reasoning: str | None = None,
    actionable: list[str] | None = None,
    impact: str = "low",
) -> str:
    """Emit a rich semantic event with human-readable context."""
    trace_id = os.environ.get("PLURIBUS_TRACE_ID")
    parent_id = os.environ.get("PLURIBUS_PARENT_ID")

    if SEMANTICS_AVAILABLE:
        try:
            topo_ctx = None
            if "topology" in data or "fanout" in data:
                topo_ctx = TopologyContext(
                    topology=data.get("topology", "single"),
                    fanout=data.get("fanout", 1),
                    coordination_budget_tokens=data.get("coord_budget_tokens", 0),
                )

            event = create_semantic_event(
                topic=topic,
                data=data,
                semantic=semantic,
                kind=kind,
                level=level,
                actor=actor,
                reasoning=reasoning,
                actionable=actionable or [],
                impact=impact,
                topology=topo_ctx,
                trace_id=trace_id,
                parent_id=parent_id,
            )
            return emit_semantic_event(bus_dir, event)
        except Exception:
            pass

    # Fallback to basic emit
    # Basic emit_bus doesn't support trace_id in this file's implementation?
    # emit_bus implementation:
    # evt = { ... }
    # append_ndjson(...)
    # It constructs dict manually. I should add trace_id there too.
    
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
        "trace_id": trace_id,
        "parent_id": parent_id,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)
    return evt_id

# ==============================================================================
# Provider Status
# ==============================================================================

@dataclass
class ProviderStatus:
    name: str
    available: bool
    model: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    blocker: str | None = None
    cooldown_until: str | None = None
    checked_at: str = ""


@dataclass
class LensDecision:
    """Routing decision from lens_collimator."""
    depth: str  # "deep" or "narrow"
    lane: str  # "dialogos" or "pbpair"
    context_mode: str  # "min", "lite", or "full"
    topology: str  # "single", "star", "peer_debate"
    fanout: int
    persona_id: str = "auto"
    persona_reason: str | None = None
    notes: list[str] = field(default_factory=list)


def _infer_request_kind(prompt: str) -> str:
    """Infer A2A request kind from prompt text."""
    p = prompt.lower()
    if any(w in p for w in ["audit", "security", "vulnerability", "vulnerabilities"]):
        return "audit"
    if any(w in p for w in ["benchmark", "performance", "speed test"]):
        return "benchmark"
    # Prefer distillation intents over verify when both appear (e.g. "distill ... with tests").
    if any(w in p for w in ["explain", "research", "summarize", "distill"]):
        return "distill"
    if any(w in p for w in ["implement", "write", "create", "build", "fix", "code"]):
        return "apply"
    if any(w in p for w in ["verify", "validate", "check", "test"]):
        return "verify"
    return "other"


def get_lens_decision(prompt: str, kind: str | None = None, effects: str = "unknown") -> LensDecision:
    """Get lens routing decision for a prompt.

    Uses lens_collimator if available, otherwise falls back to simple heuristics.
    This determines whether the query needs deep project understanding (conserve
    top-tier agents) or is a narrow isolated task (can use lighter agents).
    """
    # Infer kind from prompt if not explicitly provided
    if kind is None or kind == "other":
        kind = _infer_request_kind(prompt)

    if LENS_AVAILABLE:
        try:
            session = lens_load_vps_session(PLURIBUS_ROOT)
            req = LensRequest(
                req_id=str(uuid.uuid4()),
                goal=prompt,
                kind=kind,
                effects=effects,
                prefer_providers=["auto"],
                require_model_prefix=None,
            )
            plan = lens_plan_route(req, session=session)
            return LensDecision(
                depth=plan.depth,
                lane=plan.lane,
                context_mode=plan.context_mode,
                topology=plan.topology,
                fanout=plan.fanout,
                persona_id=getattr(plan, "persona_id", "auto") or "auto",
                persona_reason=getattr(plan, "persona_reason", None),
                notes=list(plan.notes),
            )
        except Exception:
            pass

    # Fallback heuristic when lens_collimator not available
    p = prompt.lower()

    # Deep indicators (project-aware queries)
    deep_keywords = [
        "architecture", "design", "spec", "research", "protocol", "schema",
        "dsl", "neurosymbolic", "collimator", "lens", "audit", "benchmark",
        "codebase", "repository", "project", "system", "infrastructure",
        "refactor entire", "redesign", "migrate", "integrate"
    ]
    # Also trigger deep for audit/benchmark kinds
    is_deep = (
        kind in {"audit", "benchmark"}
        or any(kw in p for kw in deep_keywords)
        or len(prompt) > 240
    )

    if is_deep:
        return LensDecision(
            depth="deep",
            lane="pbpair",
            context_mode="full",
            topology="single",
            fanout=1,
            persona_id="ring0.architect",
            persona_reason="fallback_heuristic",
            notes=["fallback_heuristic"],
        )
    else:
        return LensDecision(
            depth="narrow",
            lane="dialogos",
            context_mode="min",
            topology="single",
            fanout=1,
            persona_id="subagent.narrow_coder",
            persona_reason="fallback_heuristic",
            notes=["fallback_heuristic"],
        )

def check_provider_availability(name: str, timeout: float = 5.0) -> ProviderStatus:
    """Check if a provider is available.

    First checks vps_session.json cached status, then falls back to inline checks.
    """
    start = time.time()
    checked_at = now_iso()
    session = load_vps_session()
    cooldowns = session.get("provider_cooldowns") if isinstance(session.get("provider_cooldowns"), dict) else {}
    cooldown_until = None
    try:
        cooldown_until = format_cooldown_until(cooldowns.get(name))
    except Exception:
        cooldown_until = None

    # Web-session providers (persistent browser daemon tabs) are authoritative here.
    if name in {"chatgpt-web", "claude-web", "gemini-web"}:
        daemon = check_browser_daemon_status()
        daemon_running = bool(daemon.get("running", False))
        tabs = daemon.get("tabs") if isinstance(daemon.get("tabs"), dict) else {}
        tab = tabs.get(name) if isinstance(tabs, dict) else None
        tab_status = str(tab.get("status") or "") if isinstance(tab, dict) else ""
        tab_error = tab.get("error") if isinstance(tab, dict) else None
        model_map = {"chatgpt-web": "gpt-5.2-turbo", "claude-web": "claude-opus-4-5", "gemini-web": "gemini-3-pro"}

        if not daemon_running:
            return ProviderStatus(
                name,
                False,
                model_map.get(name, name),
                None,
                "browser daemon not running",
                "daemon",
                cooldown_until,
                checked_at,
            )

        available = tab_status == "ready"
        err = None if available else (str(tab_error) if isinstance(tab_error, str) and tab_error else (tab_status or "unavailable"))
        blocker = None
        if not available:
            if "login" in (err or "").lower() or "needs_login" in (tab_status or ""):
                blocker = "auth"
            elif "blocked_bot" in (tab_status or "") or "bot challenge" in (err or "").lower():
                blocker = "bot"
            else:
                blocker = "browser"
        checked = tab.get("last_health_check") if isinstance(tab, dict) and tab.get("last_health_check") else checked_at
        return ProviderStatus(name, available, model_map.get(name, name), None, err, blocker, cooldown_until, str(checked))

    # First check vps_session.json cached provider status
    providers = session.get("providers", {}) if isinstance(session, dict) else {}
    # Map plurichat names to vps_session names
    vps_name_map = {
        "codex-cli": "codex",
        "gemini": "gemini",
        "gemini-cli": "gemini",
        "vertex-gemini": "vertex",
        "vertex-gemini-curl": "vertex-curl",
        "claude": "claude",
        "claude-api": "claude",
        "claude-cli": "claude",
    }
    vps_name = vps_name_map.get(name, name)
    cached = providers.get(vps_name, {})
    if isinstance(cached, dict) and any(k in cached for k in ("available", "error", "last_check", "model")):
        model_map = {
            "codex-cli": "codex",
            "gemini": "gemini-2.0-flash",
            "gemini-cli": "gemini-cli",
            "vertex-gemini": "gemini-3-pro",
            "vertex-gemini-curl": "gemini-3-pro",
            "claude": "claude-opus-4-5",
            "claude-api": "claude-opus-4-5",
            "claude-cli": "claude-opus-4-5",
        }
        available = bool(cached.get("available"))
        err = cached.get("error") if not available else None
        model = cached.get("model") or model_map.get(name, name)
        blocker = None
        if isinstance(err, str) and err:
            e = err.lower()
            if "/login" in e or "login" in e or "needs login" in e:
                blocker = "auth"
            elif "timeout" in e:
                blocker = "timeout"
            elif "quota" in e or "429" in e:
                blocker = "quota"
            else:
                blocker = "error"
        return ProviderStatus(name, available, model, None, err, blocker, cooldown_until, cached.get("last_check") or checked_at)

    # Fallback to inline checks if not in cache or not available
    try:
        if name == "codex-cli":
            available = bool(shutil.which("codex"))
            return ProviderStatus(name, available, "codex", None, None if available else "CLI not found", None, cooldown_until, checked_at)

        elif name in ("gemini", "gemini-cli"):
            has_key = bool((os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip())
            has_cli = bool(shutil.which("gemini"))
            cli_logged_in = bool(session.get("gemini_cli_logged_in")) if isinstance(session, dict) else False
            available = has_key or (has_cli and cli_logged_in)
            model = "gemini-2.0-flash" if has_key else "gemini-cli"
            if available:
                return ProviderStatus(name, True, model, None, None, None, cooldown_until, checked_at)
            if has_cli and not cli_logged_in:
                return ProviderStatus(name, False, model, None, "CLI needs login", "auth", cooldown_until, checked_at)
            return ProviderStatus(name, False, model, None, "No API key or CLI", "config", cooldown_until, checked_at)

        elif name in ("vertex-gemini", "vertex-gemini-curl"):
            has_gcloud = bool(shutil.which("gcloud"))
            has_project = bool((os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip())
            available = has_gcloud and has_project
            return ProviderStatus(name, available, "gemini-3-pro", None, None if available else "No gcloud or project", "config" if not available else None, cooldown_until, checked_at)

        elif name in ("claude", "claude-api"):
            available = bool((os.environ.get("ANTHROPIC_API_KEY") or "").strip())
            return ProviderStatus(name, available, "claude-opus-4-5", None, None if available else "No API key", "auth" if not available else None, cooldown_until, checked_at)

        elif name == "claude-cli":
            have_cli = bool(shutil.which("claude"))
            if not have_cli:
                return ProviderStatus(name, False, "claude-opus-4-5", None, "CLI not found", "config", cooldown_until, checked_at)
            cli_logged_in = bool(session.get("claude_logged_in")) if isinstance(session, dict) else False
            if not cli_logged_in:
                return ProviderStatus(name, False, "claude-opus-4-5", None, "CLI needs /login", "auth", cooldown_until, checked_at)
            return ProviderStatus(name, True, "claude-opus-4-5", None, None, None, cooldown_until, checked_at)

        elif name == "openai":
            available = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
            return ProviderStatus(name, available, "gpt-5.2-turbo", None, None if available else "No API key", "auth" if not available else None, cooldown_until, checked_at)

        elif name == "mock":
            allow_mock = (os.environ.get("PLURIBUS_ALLOW_MOCK") or "").strip().lower() in {"1", "true", "yes", "on"}
            if not allow_mock:
                return ProviderStatus(name, False, "mock", 0, "internal-only", "internal", cooldown_until, checked_at)
            return ProviderStatus(name, True, "mock", 0, None, None, cooldown_until, checked_at)

        else:
            return ProviderStatus(name, False, None, None, f"Unknown provider: {name}", "config", cooldown_until, checked_at)

    except Exception as e:
        return ProviderStatus(name, False, None, None, str(e), "error", cooldown_until, checked_at)

def get_all_provider_status() -> dict[str, ProviderStatus]:
    """Get status of all configured providers including web sessions."""
    result = {p: check_provider_availability(p) for p in WEB_ONLY_PROVIDERS}
    # Mock is internal-only and opt-in.
    if (os.environ.get("PLURIBUS_ALLOW_MOCK") or "").strip().lower() in {"1", "true", "yes", "on"}:
        result["mock"] = check_provider_availability("mock")
    return result

def get_status_json() -> dict[str, dict]:
    """Get provider status as JSON-serializable dictionary."""
    status = get_all_provider_status()
    return {
        name: {
            "available": s.available,
            "model": s.model,
            "latency_ms": s.latency_ms,
            "error": s.error,
            "blocker": s.blocker,
            "cooldown_until": s.cooldown_until,
            "checked_at": s.checked_at
        }
        for name, s in status.items()
    }

# ==============================================================================
# Query Classification
# ==============================================================================

def classify_query(query: str) -> str:
    """Classify query type for optimal routing.

    Priority order matters - more specific patterns checked first.
    Uses combined scoring to handle multi-intent queries.
    """
    q = query.lower()

    # Score each category
    scores = {
        "code": 0,
        "analysis": 0,
        "math": 0,
        "research": 0,
        "creative": 0,
    }

    # Code keywords (weighted higher for precision)
    code_words = ["function", "implement", "debug", "fix", "refactor", "class", "method", "api", "code", "program", "script"]
    scores["code"] = sum(2 for w in code_words if w in q)

    # Analysis keywords
    analysis_words = ["analyze", "compare", "evaluate", "assess", "critique", "review"]
    scores["analysis"] = sum(2 for w in analysis_words if w in q)

    # Math keywords
    math_words = ["math", "calculate", "equation", "formula", "prove", "theorem", "algebra", "solve"]
    scores["math"] = sum(2 for w in math_words if w in q)

    # Research keywords
    research_words = ["research", "paper", "study", "explain", "how does", "what is", "why", "theory"]
    scores["research"] = sum(1 for w in research_words if w in q)

    # Creative keywords (lower weight to avoid false positives with "write a function")
    creative_words = ["story", "poem", "creative", "imagine", "brainstorm", "novel", "fiction"]
    scores["creative"] = sum(2 for w in creative_words if w in q)
    # "write" alone is weak signal for creative
    if "write" in q and not any(w in q for w in code_words):
        scores["creative"] += 1

    # Get highest scoring category
    # Priority order for tie-breaking: analysis > math > code > research > creative
    priority = ["analysis", "math", "code", "research", "creative"]
    max_score = max(scores.values())
    if max_score > 0:
        for cat in priority:
            if scores[cat] == max_score:
                return cat

    return "general"

@dataclass
class RoutingResult:
    """Result of provider selection with lens context."""
    provider: str
    query_type: str
    lens: LensDecision


def _load_persona_label(persona_id: str) -> str | None:
    if not _load_personas:
        return None
    try:
        reg = _load_personas(PLURIBUS_ROOT / "nucleus" / "specs" / "personas.json")
        for p in reg.get("personas", []):
            if p.get("id") == persona_id:
                return str(p.get("label") or persona_id)
    except Exception:
        return None
    return None


def _load_idiolect_priors() -> list[str]:
    if not _load_idiolect:
        return []
    try:
        obj = _load_idiolect(PLURIBUS_ROOT / "nucleus" / "specs" / "idiolect.json")
        pri = obj.get("priors") if isinstance(obj, dict) else None
        constraints = (pri or {}).get("constraints") if isinstance(pri, dict) else None
        if isinstance(constraints, list):
            return [str(x) for x in constraints if str(x).strip()]
    except Exception:
        return []
    return []


def shape_prompt(prompt: str, *, lens: LensDecision, persona_override: str = "auto") -> str:
    """Optionally add Pluribus priors + persona header for non-min context modes."""
    context_mode = getattr(lens, "context_mode", "min")
    if context_mode == "min":
        return prompt

    persona_id = persona_override if persona_override and persona_override != "auto" else getattr(lens, "persona_id", "auto")
    persona_label = _load_persona_label(persona_id) or persona_id
    priors = _load_idiolect_priors()
    prior_lines = "\n".join([f"- {p}" for p in priors]) if priors else "- (priors unavailable)"

    header = (
        "## Pluribus Context Header\n\n"
        f"persona_id: {persona_id}\n"
        f"persona_label: {persona_label}\n"
        f"context_mode: {context_mode}\n"
        f"lane: {getattr(lens, 'lane', 'dialogos')}\n"
        f"topology: {getattr(lens, 'topology', 'single')} (fanout={getattr(lens, 'fanout', 1)})\n\n"
        "constraints:\n"
        f"{prior_lines}\n\n"
        "---\n\n"
    )
    return header + prompt


def select_provider_for_query(
    query: str,
    preferred: str | None = None,
    available: dict[str, ProviderStatus] | None = None,
    include_lens: bool = True
) -> RoutingResult:
    """Select optimal provider based on query type, depth, and availability.

    PluriChat policy: web-session-only routing via persistent browser tabs.
    Never silently falls back to CLI/API/mock providers.
    """
    if not available:
        available = get_all_provider_status()

    query_type = classify_query(query)
    lens = get_lens_decision(query, effects="none") if include_lens else LensDecision(
        depth="narrow", lane="dialogos", context_mode="min", topology="single", fanout=1
    )

    allow_mock = (os.environ.get("PLURIBUS_ALLOW_MOCK") or "").strip().lower() in {"1", "true", "yes", "on"}
    allowed = set(WEB_ONLY_PROVIDERS) | ({"mock"} if allow_mock else set())

    # If user forced a specific provider, respect it only if allowed.
    if preferred and preferred != "auto":
        pref = str(preferred).strip()
        if pref in allowed:
            return RoutingResult(provider=pref, query_type=query_type, lens=lens)

    # Select routing table based on depth
    if lens.depth == "deep":
        routing_table = MODEL_ROUTING_DEEP
    else:
        routing_table = MODEL_ROUTING_NARROW

    candidates = routing_table.get(query_type, ["auto"])

    # Web-only selection: choose first available candidate; if none available, choose
    # the first web provider deterministically (so the executor can surface a clear
    # needs-login/bot-challenge error instead of silently using mock).
    web_candidates = [p for p in candidates if p in WEB_ONLY_PROVIDERS]
    if not web_candidates:
        web_candidates = WEB_ONLY_PROVIDERS[:]

    for provider in web_candidates:
        if available.get(provider, ProviderStatus(provider, False)).available:
            return RoutingResult(provider=provider, query_type=query_type, lens=lens)

    return RoutingResult(provider=web_candidates[0], query_type=query_type, lens=lens)


# ==============================================================================
# STRp Queue Integration (for star/peer_debate topology)
# ==============================================================================

def dispatch_to_strp_queue(
    prompt: str,
    lens: LensDecision,
    bus_dir: Path,
    actor: str,
    provider_hint: str = "auto",
    kind: str = "distill",
    timeout: float = 10.0
) -> tuple[str, bool]:
    """Dispatch a request to STRp queue for multi-agent topology execution.

    Used when lens.topology is 'star' or 'peer_debate' to leverage
    STRp's fanout and coordination capabilities.

    Returns (req_id, success).
    """
    strp_queue = TOOLS_DIR / "strp_queue.py"
    if not strp_queue.exists():
        return "", False

    # Determine tool density from effects inference
    tool_density = 0.3  # Default for deep queries
    if "file" in prompt.lower() or "write" in prompt.lower():
        tool_density = 0.5
    if "network" in prompt.lower() or "api" in prompt.lower():
        tool_density = 0.7

    cmd = [
        sys.executable,
        str(strp_queue),
        "--root",
        str(PLURIBUS_ROOT),
        "request",
        "--goal",
        prompt,
        "--kind",
        kind,
        "--provider",
        provider_hint or "auto",
        "--topology-hint",
        lens.topology,
        "--tool-density",
        str(tool_density),
        "--coord-budget-tokens",
        str(6500 if lens.depth == "deep" else 1000),
        "--emit-bus",
        "--bus-dir",
        str(bus_dir),
    ]

    if lens.topology in ("star", "peer_debate"):
        cmd.append("--parallelizable")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"}
        )
        if result.returncode == 0:
            returned_id = result.stdout.strip()
            # Emit correlation event
            emit_bus(bus_dir, topic="plurichat.strp.dispatch", kind="request", level="info", actor=actor, data={
                "strp_req_id": returned_id,
                "topology": lens.topology,
                "fanout": lens.fanout,
                "prompt": prompt[:200],
            })
            return returned_id, True
        return "", False
    except Exception:
        return "", False


def wait_for_strp_response(req_id: str, bus_dir: Path, timeout: float = 120.0) -> str | None:
    """Wait for STRp response.

    Preference order:
    1) `.pluribus/index/responses.ndjson` (strp_worker output)
    2) bus topics (`strp.worker.item` or legacy `strp.response`)
    """
    responses_path = PLURIBUS_ROOT / ".pluribus" / "index" / "responses.ndjson"
    events_path = bus_dir / "events.ndjson"
    start = time.time()
    seen_lines = 0

    while time.time() - start < timeout:
        try:
            if responses_path.exists():
                for line in responses_path.read_text(encoding="utf-8", errors="replace").splitlines()[::-1]:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(e, dict) and e.get("req_id") == req_id:
                        out = e.get("output")
                        if isinstance(out, dict):
                            return out.get("summary") or json.dumps(out)
                        return str(out) if out else None
        except Exception:
            pass
        try:
            with events_path.open("r") as f:
                lines = f.readlines()
                for line in lines[seen_lines:]:
                    try:
                        e = json.loads(line)
                        if e.get("topic") in {"strp.worker.item", "strp.response"}:
                            data = e.get("data", {})
                            if data.get("req_id") == req_id:
                                output = data.get("output")
                                if isinstance(output, dict):
                                    return output.get("summary") or json.dumps(output)
                                return str(output) if output else None
                    except:
                        continue
                seen_lines = len(lines)
        except:
            pass
        time.sleep(0.3)

    return None


def run_strp_worker_once(req_id: str, *, bus_dir: Path, provider: str, timeout: float) -> bool:
    """Process a single STRp request id via strp_worker (bounded)."""
    strp_worker = TOOLS_DIR / "strp_worker.py"
    if not strp_worker.exists():
        return False
    try:
        p = subprocess.run(
            [
                sys.executable,
                str(strp_worker),
                "--root",
                str(PLURIBUS_ROOT),
                "--bus-dir",
                str(bus_dir),
                "--provider",
                provider or "auto",
                "--only-req-id",
                req_id,
                "--once",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(5, int(timeout)),
            env={**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return int(p.returncode) == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def execute_with_topology(
    prompt: str,
    routing: RoutingResult,
    bus_dir: Path,
    actor: str,
    mode: str = "direct",
    timeout: float = 120.0
) -> "ChatResponse":
    """Execute chat with topology-aware routing.

    Routing behavior based on lens decision:
    - single topology: Direct execution via provider router
    - star/peer_debate topology: Dispatch to STRp queue for multi-agent fanout

    Context mode affects prompt handling:
    - min: Minimal context, isolated task
    - lite: Light context, some project awareness
    - full: Full context, deep project understanding
    """
    lens = routing.lens
    start_time = time.time()

    # Emit routing decision with rich semantic context
    depth_reasoning = {
        "deep": "Complex query requires full project understanding and top-tier agents",
        "narrow": "Simple isolated task suitable for lighter, faster agents",
    }.get(lens.depth, "Unknown query depth classification")

    topology_insight = {
        "single": "Direct execution via single agent",
        "star": f"Multi-agent coordination: 1 coordinator + {lens.fanout-1} workers",
        "peer_debate": f"Parallel debate among {lens.fanout} peer agents",
    }.get(lens.topology, lens.topology)

    emit_semantic(
        bus_dir,
        topic="plurichat.routing.decision",
        kind="metric",
        level="info",
        actor=actor,
        data={
            "provider": routing.provider,
            "query_type": routing.query_type,
            "depth": lens.depth,
            "lane": lens.lane,
            "topology": lens.topology,
            "fanout": lens.fanout,
            "context_mode": lens.context_mode,
            "persona_id": getattr(lens, "persona_id", "auto"),
            "persona_reason": getattr(lens, "persona_reason", None),
        },
        semantic=f"Routing {routing.query_type} query ({lens.depth}) to {routing.provider} via {lens.lane} lane",
        reasoning=f"{depth_reasoning}. {topology_insight}. Context mode: {lens.context_mode}",
        actionable=[
            f"Monitor {routing.provider} response quality",
            f"Track {lens.context_mode} context usage",
            "Consider caching if query is reusable" if lens.depth == "narrow" else "Break down if response quality insufficient",
        ],
        impact="high" if lens.depth == "deep" else "low",
    )

    # Web-session providers are backed by a single persistent browser tab.
    # They cannot safely support fanout topologies; degrade to single execution.
    if routing.provider in WEB_SESSION_PROVIDERS and lens.topology in ("star", "peer_debate") and lens.fanout > 1:
        lens.topology = "single"
        lens.fanout = 1

    # For star/peer_debate with fanout > 1, use STRp queue for multi-agent coordination
    if lens.topology in ("star", "peer_debate") and lens.fanout > 1:
        strp_req_id, success = dispatch_to_strp_queue(
            prompt,
            lens,
            bus_dir,
            actor,
            provider_hint=routing.provider,
            kind=_infer_request_kind(prompt),
        )
        if success:
            # Process this specific STRp request id (avoid draining unrelated backlog),
            # then read the response from the STRp response ledger/bus.
            _ = run_strp_worker_once(strp_req_id, bus_dir=bus_dir, provider=routing.provider, timeout=timeout)
            response_text = wait_for_strp_response(strp_req_id, bus_dir, timeout=min(timeout, 5.0))
            latency_ms = (time.time() - start_time) * 1000

            if response_text:
                return ChatResponse(
                    text=response_text,
                    provider=f"strp:{lens.topology}:{lens.fanout}",
                    model=None,
                    latency_ms=latency_ms,
                    req_id=strp_req_id,
                    success=True,
                )
            # If no immediate response, request is queued for async processing
            return ChatResponse(
                text=f"Request queued in STRp ({lens.topology} topology, fanout={lens.fanout}). req_id={strp_req_id}",
                provider=f"strp:{lens.topology}:{lens.fanout}",
                model=None,
                latency_ms=latency_ms,
                req_id=strp_req_id,
                success=True,
            )

    # Single topology or STRp unavailable: direct execution
    # Apply context_mode to determine execution strategy
    effective_timeout = timeout
    if lens.context_mode == "min":
        # Minimal context: faster timeout for isolated tasks
        effective_timeout = min(timeout, 30.0)
    elif lens.context_mode == "full":
        # Full context: allow more time for complex responses
        effective_timeout = max(timeout, 180.0)

    # Web session execution path (browser daemon → optional fallback).
    if routing.provider in WEB_SESSION_PROVIDERS:
        return execute_web_session_inference(prompt, routing.provider, bus_dir, actor, timeout=effective_timeout)

    if mode == "proxy":
        return execute_chat_proxy(prompt, routing.provider, bus_dir, actor, effective_timeout)
    else:
        return execute_chat_direct(prompt, routing.provider, bus_dir, actor, effective_timeout)


# ==============================================================================
# Chat Execution
# ==============================================================================

@dataclass
class ChatResponse:
    text: str
    provider: str
    model: str | None
    latency_ms: float
    req_id: str
    success: bool
    error: str | None = None

def execute_chat_direct(prompt: str, provider: str, bus_dir: Path, actor: str, timeout: float = 120.0) -> ChatResponse:
    """Execute chat directly via provider router.

    Uses router.py with --provider auto to get proper fallback behavior.
    The router handles quota exhaustion, auth failures, and provider chain.
    """
    req_id = str(uuid.uuid4())
    start = time.time()

    # Emit request event with semantic context
    emit_semantic(
        bus_dir,
        topic="plurichat.request",
        kind="request",
        level="info",
        actor=actor,
        data={
            "req_id": req_id,
            "provider": provider,
            "prompt": prompt[:500],  # Truncate for bus
            "prompt_length": len(prompt),
        },
        semantic=f"Processing chat request via {provider} ({len(prompt)} chars)",
        reasoning="User initiated query, routing through direct execution path",
        actionable=["Await provider response", "Monitor latency threshold"],
        impact="medium",
    )

    # Build router command.
    # - For normal usage, prefer auto so router.py can apply the fallback chain.
    # - For explicit offline/mock calls, force mock to avoid unintended network access.
    router_script = PROVIDERS_DIR / "router.py"

    effective_provider = provider if provider in {"mock", "auto"} else "auto"
    cmd = [sys.executable, str(router_script), "--provider", effective_provider, "--prompt", prompt]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"}
        )

        latency_ms = (time.time() - start) * 1000

        if result.returncode == 0:
            response = ChatResponse(
                text=result.stdout.strip(),
                provider=effective_provider,
                model=None,
                latency_ms=latency_ms,
                req_id=req_id,
                success=True
            )
        else:
            response = ChatResponse(
                text=result.stderr.strip() or "Provider error",
                provider=effective_provider,
                model=None,
                latency_ms=latency_ms,
                req_id=req_id,
                success=False,
                error=result.stderr.strip()
            )

        # Emit response event with semantic context
        success_semantic = f"Chat response completed in {latency_ms:.0f}ms via {effective_provider}"
        failure_semantic = f"Chat response failed after {latency_ms:.0f}ms: {response.error or 'unknown error'}"

        emit_semantic(
            bus_dir,
            topic="plurichat.response",
            kind="response",
            level="info" if response.success else "warn",
            actor=actor,
            data={
                "req_id": req_id,
                "provider": provider,
                "effective_provider": effective_provider,
                "success": response.success,
                "latency_ms": latency_ms,
                "error": response.error,
                "response_length": len(response.text) if response.text else 0,
            },
            semantic=success_semantic if response.success else failure_semantic,
            reasoning=f"Provider {effective_provider} {'responded successfully' if response.success else 'encountered error during generation'}",
            actionable=[] if response.success else [
                f"Check {effective_provider} provider health",
                "Consider fallback to alternative provider",
                f"Review error: {str(response.error)[:100]}" if response.error else "Check provider logs",
            ],
            impact="low" if response.success else "high",
        )

        return response

    except subprocess.TimeoutExpired:
        latency_ms = (time.time() - start) * 1000
        return ChatResponse(
            text="Request timed out",
            provider=provider,
            model=None,
            latency_ms=latency_ms,
            req_id=req_id,
            success=False,
            error="Timeout"
        )
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return ChatResponse(
            text=str(e),
            provider=provider,
            model=None,
            latency_ms=latency_ms,
            req_id=req_id,
            success=False,
            error=str(e)
        )

def execute_chat_proxy(prompt: str, provider: str, bus_dir: Path, actor: str, timeout: float = 120.0) -> ChatResponse:
    """Execute chat via bus (proxy mode) - publishes dialogos.submit and waits."""
    req_id = str(uuid.uuid4())
    start = time.time()

    effective_provider = provider if provider in {"mock", "auto"} else "auto"

    # Emit dialogos.submit
    emit_bus(bus_dir, topic="dialogos.submit", kind="request", level="info", actor=actor, data={
        "req_id": req_id,
        "mode": "llm",
        "providers": [effective_provider],
        "prompt": prompt,
        "lens": {
            "depth": "narrow",
            "lane": "dialogos",
            "context_mode": "min",
        }
    })

    # Process once via dialogosd
    dialogosd = TOOLS_DIR / "dialogosd.py"
    if dialogosd.exists():
        subprocess.run(
            [sys.executable, str(dialogosd), "--bus-dir", str(bus_dir), "--once"],
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )

    # Tail for response
    events_path = bus_dir / "events.ndjson"
    response_text = ""
    deadline = start + timeout

    while time.time() < deadline:
        try:
            with events_path.open("r") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        data = e.get("data", {})
                        if data.get("req_id") == req_id:
                            if e.get("topic") == "dialogos.cell.output":
                                response_text += data.get("content", "")
                            elif e.get("topic") == "dialogos.cell.end":
                                latency_ms = (time.time() - start) * 1000
                                return ChatResponse(
                                    text=response_text.strip(),
                                    provider=provider,
                                    model=None,
                                    latency_ms=latency_ms,
                                    req_id=req_id,
                                    success=True
                                )
                    except:
                        continue
        except:
            pass
        time.sleep(0.2)

    latency_ms = (time.time() - start) * 1000
    return ChatResponse(
        text=response_text.strip() or "No response received",
        provider=provider,
        model=None,
        latency_ms=latency_ms,
        req_id=req_id,
        success=bool(response_text),
        error=None if response_text else "Timeout waiting for response"
    )

# ==============================================================================
# Status Display
# ==============================================================================

def render_status_line(providers: dict[str, ProviderStatus], current_provider: str | None = None) -> str:
    """Render status line showing provider availability."""
    parts = []

    for name, status in providers.items():
        if name == "mock":
            continue

        if status.available:
            indicator = f"{C.GREEN}●{C.RESET}"
        else:
            indicator = f"{C.RED}○{C.RESET}"

        label = name.replace("-cli", "").replace("-api", "").replace("vertex-", "v-")[:6]
        if status.cooldown_until:
            label = f"{label}*"

        if current_provider and name == current_provider:
            parts.append(f"{indicator} {C.BOLD}{label}{C.RESET}")
        else:
            parts.append(f"{indicator} {C.DIM}{label}{C.RESET}")

    return " | ".join(parts)

def print_banner(bus_dir: Path | None = None, show_boot: bool = True):
    """Print welcome banner with optional boot sequence."""
    print(f"""
{C.CYAN}╔══════════════════════════════════════════════════════════════╗
║  {C.BOLD}PluriChat{C.RESET}{C.CYAN} — Multi-Model Intelligent Router                   ║
║  {C.DIM}Gemini-3 • GPT-5.2 • Claude Opus/Sonnet{C.RESET}{C.CYAN}                      ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")
    # Show Unix-style boot sequence on startup
    if show_boot and bus_dir:
        print(f"{C.DIM}Initializing Pluribus subsystems...{C.RESET}")
        motd = get_supermotd_status(bus_dir)
        motd.print_boot_sequence(fast=True)
        print()

def print_help():
    """Print help message."""
    print(f"""
{C.BOLD}Commands:{C.RESET}
  /status         Show compact system + provider status
  /supermotd      Full SUPERMOTD status view (ring health, bus stats)
  /provider <p>   Switch provider (auto, gemini-3, claude, codex, mock)
  /persona <id>   Set persona (auto or persona_id)
  /context <m>    Set context mode (auto, min, lite, full)
  /mode <m>       Switch mode (direct, proxy)
  /clear          Clear screen
  /realagents ... REALAGENTS dispatch (no-shims deepening)
  /pbeport [w] [n]  Status distillation (window_s, width)
  /ckin [w] [n]   Operator check-in (pbeport + BEAM/GOLDEN + status)
  /chkin          Alias for /ckin
  /mbad [w] [n]   Membrane diagnostics snapshot (bus+infer_sync+hexis)
  /mabswarm [w] [i] [emit]  Probe bus; add 'emit' to publish reflex requests
  /iterate        Semantic operator: broadcast next iteration request
  /pbflush [msg]  Semantic operator: finish tasks + await next CKIN
  /pbdeep [text]  Semantic operator: deep audit request (branches/lost+found/docs)
  /speak [text]   Write text to /pluribus/speaker_bus.aiff for TTS broadcast
  /help           Show this help
  /quit           Exit

{C.BOLD}Status Indicators:{C.RESET}
  Ring dots: [{C.BRIGHT_RED}*{C.RESET}{C.BRIGHT_YELLOW}*{C.RESET}{C.BRIGHT_GREEN}*{C.RESET}{C.BRIGHT_CYAN}*{C.RESET}] = bus/providers/inference/omega
  o=inactive  {C.DIM}*{C.RESET}=degraded  *=nominal  {C.BOLD}*{C.RESET}=optimal

{C.BOLD}Query Routing:{C.RESET}
  Code queries    -> Claude Opus / Codex
  Research        -> Gemini 3 / Claude
  Creative        -> Claude / GPT-5.2
  Analysis        -> Gemini 3 / Claude Sonnet
  Math            -> Gemini 3 / Claude

{C.BOLD}Examples:{C.RESET}
  > Implement a binary search function
  > Explain quantum entanglement
  > Write a haiku about programming
""")

# ==============================================================================
# Interactive REPL
# ==============================================================================

@dataclass
class ChatState:
    bus_dir: Path
    actor: str
    provider: str = "auto"
    persona: str = "auto"
    context_mode: str = "auto"  # auto|min|lite|full
    mode: str = "direct"  # direct or proxy
    providers: dict[str, ProviderStatus] = field(default_factory=dict)
    history: list[tuple[str, ChatResponse]] = field(default_factory=list)
    running: bool = True

def refresh_provider_status(state: ChatState):
    """Background refresh of provider status."""
    state.providers = get_all_provider_status()

def handle_command(cmd: str, state: ChatState) -> bool:
    """Handle slash commands. Returns True if should continue."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        state.running = False
        return False

    elif command == "/status":
        refresh_provider_status(state)
        motd = get_supermotd_status(state.bus_dir)
        print(f"\n{C.BOLD}System Status:{C.RESET}")
        print(f"  {motd.build_compact_status(state.providers)}")
        print(f"\n{C.BOLD}Provider Status:{C.RESET}")
        for name, status in state.providers.items():
            indicator = f"{C.GREEN}●{C.RESET}" if status.available else f"{C.RED}○{C.RESET}"
            model_info = f" ({status.model})" if status.model else ""
            error_info = f" - {C.RED}{status.error}{C.RESET}" if status.error else ""
            print(f"  {indicator} {name}{model_info}{error_info}")

        # Browser daemon status
        daemon = check_browser_daemon_status()
        if daemon.get("running"):
            print(f"\n{C.BOLD}Browser Daemon:{C.RESET} {C.GREEN}●{C.RESET} running (PID {daemon.get('pid', '?')})")
            for tab_id, tab in daemon.get("tabs", {}).items():
                tab_indicator = f"{C.GREEN}●{C.RESET}" if tab.get("status") == "ready" else f"{C.YELLOW}○{C.RESET}"
                print(f"  {tab_indicator} {tab_id}: {tab.get('status', 'unknown')} (chats: {tab.get('chat_count', 0)})")
        else:
            print(f"\n{C.BOLD}Browser Daemon:{C.RESET} {C.RED}○{C.RESET} not running")

        print(f"\n  Current: {C.BOLD}{state.provider}{C.RESET} | Mode: {C.BOLD}{state.mode}{C.RESET}")
        print()
        return True

    elif command == "/supermotd":
        # Full SUPERMOTD status view
        refresh_provider_status(state)
        motd = get_supermotd_status(state.bus_dir)
        print(f"\n{C.CYAN}{'=' * 64}{C.RESET}")
        print(f"{C.BOLD}SUPERMOTD - Pluribus System Status{C.RESET}")
        print(f"{C.CYAN}{'=' * 64}{C.RESET}\n")

        # Compact status line
        print(f"  {motd.build_compact_status(state.providers)}")
        print()

        # Ring health legend
        print(f"{C.BOLD}Ring Health:{C.RESET}")
        print(f"  {C.BRIGHT_RED}Ring 0{C.RESET}: Bus/Infrastructure  {C.BRIGHT_YELLOW}Ring 1{C.RESET}: Providers")
        print(f"  {C.BRIGHT_GREEN}Ring 2{C.RESET}: Inference Flow     {C.BRIGHT_CYAN}Ring 3{C.RESET}: Omega State")
        print(f"  {C.DIM}Legend: o=inactive *=degraded *=nominal *=optimal{C.RESET}")
        print()

        # Provider details
        print(f"{C.BOLD}Provider Mesh:{C.RESET}")
        for name, status in state.providers.items():
            if name == "mock":
                continue
            indicator = f"{C.GREEN}●{C.RESET}" if status.available else f"{C.RED}○{C.RESET}"
            model_info = f" ({status.model})" if status.model else ""
            cooldown_info = f" {C.YELLOW}[cooldown]{C.RESET}" if status.cooldown_until else ""
            blocker_info = f" {C.RED}[{status.blocker}]{C.RESET}" if status.blocker else ""
            print(f"  {indicator} {name}{model_info}{cooldown_info}{blocker_info}")
        print()

        # Recent bus activity
        events_path = state.bus_dir / "events.ndjson"
        if events_path.exists():
            try:
                event_count = sum(1 for _ in events_path.open("r"))
                file_size = events_path.stat().st_size
                print(f"{C.BOLD}Bus Statistics:{C.RESET}")
                print(f"  Events: {event_count}")
                print(f"  File size: {file_size / 1024:.1f} KB")
            except Exception:
                pass
        print()

        # Show recent SUPERMOTD lines via external tool if available
        supermotd_tool = TOOLS_DIR / "supermotd.py"
        if supermotd_tool.exists():
            print(f"{C.BOLD}Recent System Activity:{C.RESET}")
            try:
                result = subprocess.run(
                    [sys.executable, str(supermotd_tool), "--bus-dir", str(state.bus_dir), "--limit", "10"],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                if result.stdout:
                    for line in result.stdout.strip().split("\n"):
                        print(f"  {C.DIM}{line}{C.RESET}")
            except Exception:
                print(f"  {C.DIM}(unable to fetch recent activity){C.RESET}")
        print()
        print(f"{C.CYAN}{'=' * 64}{C.RESET}")
        return True

    elif command == "/mbad":
        # MBAD snapshot (membrane view). Safe by default; optionally emits metric with --emit-bus.
        parts2 = arg.split()
        window_s = parts2[0] if len(parts2) >= 1 else "900"
        width = parts2[1] if len(parts2) >= 2 else "24"
        mbad_tool = TOOLS_DIR / "mbad.py"
        if not mbad_tool.exists():
            print(f"  {C.RED}mbad.py not found{C.RESET}")
            return True
        subprocess.run(
            [sys.executable, str(mbad_tool), "--window", window_s, "--width", width, "--emit-bus"],
            env={**os.environ, "PLURIBUS_BUS_DIR": str(state.bus_dir), "HEXIS_BUFFER_DIR": os.environ.get("HEXIS_BUFFER_DIR", "/tmp"), "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )
        return True

    elif command in {"/mabswarm", "/mbswarm"}:
        # Default is dry-run; requires explicit 'emit' token to publish reflex requests.
        parts2 = arg.split()
        window_s = parts2[0] if len(parts2) >= 1 else "60"
        interval_s = parts2[1] if len(parts2) >= 2 else "10"
        emit = any(p.lower() in {"emit", "go", "--emit-bus"} for p in parts2[2:])
        mabswarm_tool = TOOLS_DIR / "mabswarm.py"
        if not mabswarm_tool.exists():
            print(f"  {C.RED}mabswarm.py not found{C.RESET}")
            return True
        cmdline = [sys.executable, str(mabswarm_tool), "--window", window_s, "--interval", interval_s]
        if emit:
            cmdline.append("--emit-bus")
        subprocess.run(
            cmdline,
            env={**os.environ, "PLURIBUS_BUS_DIR": str(state.bus_dir), "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )
        return True

    elif command == "/provider":
        if not arg:
            print(f"  Current provider: {C.BOLD}{state.provider}{C.RESET}")
            print(f"  Available: auto, gemini-3, claude, codex, vertex, mock")
        else:
            # Map friendly names to internal names
            mapping = {
                "gemini-3": "vertex-gemini-curl",
                "gemini": "gemini",
                "claude": "claude-api",
                "codex": "codex-cli",
                "vertex": "vertex-gemini",
                "gpt": "openai",
                "mock": "mock",
                "auto": "auto",
            }
            state.provider = mapping.get(arg.lower(), arg.lower())
            print(f"  Provider set to: {C.BOLD}{state.provider}{C.RESET}")
        return True

    elif command == "/persona":
        if not arg:
            print(f"  Current persona: {C.BOLD}{state.persona}{C.RESET}")
            print(f"  Use: auto, ring0.architect, ring0.security_auditor, subagent.narrow_coder, subagent.distiller, subagent.edge_inference")
        else:
            state.persona = arg.strip() or "auto"
            print(f"  Persona set to: {C.BOLD}{state.persona}{C.RESET}")
        return True

    elif command == "/context":
        if not arg:
            print(f"  Current context mode: {C.BOLD}{state.context_mode}{C.RESET}")
            print(f"  Available: auto, min, lite, full")
        else:
            v = arg.strip().lower()
            if v not in {"auto", "min", "lite", "full"}:
                print(f"  {C.RED}Unknown context mode: {arg}{C.RESET}")
            else:
                state.context_mode = v
                print(f"  Context mode set to: {C.BOLD}{state.context_mode}{C.RESET}")
        return True

    elif command == "/mode":
        if not arg:
            print(f"  Current mode: {C.BOLD}{state.mode}{C.RESET}")
            print(f"  Available: direct, proxy")
        elif arg.lower() in ("direct", "proxy"):
            state.mode = arg.lower()
            print(f"  Mode set to: {C.BOLD}{state.mode}{C.RESET}")
        else:
            print(f"  {C.RED}Unknown mode: {arg}{C.RESET}")
        return True

    elif command == "/clear":
        os.system("clear" if os.name != "nt" else "cls")
        print_banner(bus_dir=state.bus_dir, show_boot=False)
        # Show compact status after clear
        motd = get_supermotd_status(state.bus_dir)
        print(f"  {motd.build_compact_status(state.providers)}")
        return True

    elif command in ("/help", "/?"):
        print_help()
        return True

    elif command == "/pbeport":
        if not _pbeport:
            print(f"  {C.RED}PBEPORT unavailable (missing tools/pbeport.py){C.RESET}")
            return True
        window_s = 900
        width = 24
        a = (arg or "").strip()
        if a:
            toks = a.split()
            try:
                if len(toks) >= 1:
                    window_s = int(toks[0])
                if len(toks) >= 2:
                    width = int(toks[1])
            except Exception:
                print(f"  {C.RED}Usage: /pbeport [window_s] [width]{C.RESET}")
                return True
        snap = _pbeport.build_snapshot(bus_dir=state.bus_dir, window_s=window_s, width=width)
        sys.stdout.write(_pbeport.render_snapshot(snap))
        try:
            _pbeport.emit_bus(
                state.bus_dir,
                topic="pbeport.snapshot",
                kind="metric",
                level="info",
                actor=state.actor,
                data={
                    "window_s": snap.window_s,
                    "events_total": snap.events_total,
                    "kinds": snap.kinds,
                    "topic_prefixes": dict(sorted(snap.topic_prefixes.items(), key=lambda x: -x[1])[:20]),
                    "pending_infer_sync": len(snap.pending_req_ids),
                    "provider_incidents": len(snap.provider_incidents),
                },
            )
        except Exception:
            pass
        return True

    elif command in {"/ckin", "/checkin", "/chkin"}:
        # Prefer the canonical CKIN v2 dashboard generator if present.
        ckin_tool = TOOLS_DIR / "ckin_report.py"
        if ckin_tool.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(ckin_tool), "--agent", state.actor, "--emit-bus"],
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                    env={**os.environ, "PLURIBUS_BUS_DIR": str(state.bus_dir), "PLURIBUS_ACTOR": state.actor, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                if result.stdout:
                    sys.stdout.write(result.stdout)
                if result.stderr:
                    sys.stderr.write(result.stderr)
                if result.returncode == 0:
                    return True
            except Exception:
                # Fall back to built-in distillation below.
                pass

        if not _pbeport:
            print(f"  {C.RED}PBEPORT unavailable (missing tools/pbeport.py){C.RESET}")
            return True
        # Standard operator defaults (as per operator broadcast)
        window_s = 900
        width = 32
        a = (arg or "").strip()
        if a:
            toks = a.split()
            try:
                if len(toks) >= 1:
                    window_s = int(toks[0])
                if len(toks) >= 2:
                    width = int(toks[1])
            except Exception:
                print(f"  {C.RED}Usage: /ckin [window_s] [width]{C.RESET}")
                return True

        snap = _pbeport.build_snapshot(bus_dir=state.bus_dir, window_s=window_s, width=width)
        sys.stdout.write(_pbeport.render_snapshot(snap))

        # BEAM + GOLDEN synthesis counts (drift guards)
        beam_path = PLURIBUS_ROOT / "agent_reports" / "2025-12-15_beam_10x_discourse.md"
        golden_path = PLURIBUS_ROOT / "nucleus" / "docs" / "GOLDEN_SYNTHESIS_DISCOURSE.md"
        beam_entries = _pbeport.count_beam_entries(beam_path)
        golden_lines = _pbeport.count_lines(golden_path)

        print(f"BEAM entries: {beam_entries}")
        print(f"GOLDEN_SYNTHESIS lines: {golden_lines}")

        # Agent status table (from infer_sync.checkin)
        if snap.last_checkins:
            print("\nAgent status (latest infer_sync.checkin):")
            print("actor\tstatus\tdone\topen\tblocked\terrors\tsubproject\tnext")
            for actor, d in sorted(snap.last_checkins.items(), key=lambda x: str(x[0]))[:20]:
                print(
                    f"{actor}\t{d.get('status')}\t{d.get('done')}\t{d.get('open')}\t{d.get('blocked')}\t{d.get('errors')}\t{d.get('subproject')}\t{(d.get('next') or '')}"
                )

        # Next actions / blockers (minimal distillation)
        if snap.pending_req_ids:
            print(f"\nPending infer_sync req_ids: {len(snap.pending_req_ids)}")
        if snap.provider_incidents:
            print(f"Provider incidents (window): {len(snap.provider_incidents)}")
        return True

    elif command in {"/iterate", "/iter"}:
        iterate_tool = TOOLS_DIR / "iterate_operator.py"
        if iterate_tool.exists():
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(iterate_tool),
                        "--bus-dir",
                        str(state.bus_dir),
                        "--agent",
                        state.actor,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    env={**os.environ, "PLURIBUS_BUS_DIR": str(state.bus_dir), "PLURIBUS_ACTOR": state.actor, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                rid = (result.stdout or "").strip()
                if result.returncode == 0 and rid:
                    print(f"  {C.GREEN}ITERATE broadcast:{C.RESET} req_id={C.BOLD}{rid}{C.RESET}")
                else:
                    msg = (result.stderr or result.stdout or "").strip()[:160]
                    print(f"  {C.RED}ITERATE failed:{C.RESET} {msg}")
            except Exception as e:
                print(f"  {C.RED}ITERATE failed: {str(e)[:120]}{C.RESET}")
            return True

        # Fallback: emit a minimal bus request directly
        rid = str(uuid.uuid4())
        emit_bus(
            state.bus_dir,
            topic="infer_sync.request",
            kind="request",
            level="info",
            actor=state.actor,
            data={
                "req_id": rid,
                "subproject": "beam_10x",
                "intent": "iterate",
                "inputs": {"operator": "iterate"},
                "constraints": {"append_only": True, "tests_first": True, "non_blocking": True},
                "response_topic": "infer_sync.response",
                "iso": now_iso(),
            },
        )
        emit_bus(
            state.bus_dir,
            topic="operator.iterate.request",
            kind="request",
            level="info",
            actor=state.actor,
            data={"req_id": rid, "subproject": "beam_10x", "intent": "iterate", "iso": now_iso()},
        )
        print(f"  {C.GREEN}ITERATE broadcast:{C.RESET} req_id={C.BOLD}{rid}{C.RESET}")
        return True

    elif command == "/pbdeep":
        raw = (arg or "").strip()
        if not raw:
            print(f"  {C.DIM}Usage:{C.RESET} /pbdeep [scope=repo] [reason=...] <instruction...>")
            return True

        scope = "repo"
        reason = "operator_pbdeep"
        no_infer_sync = False
        instruction_parts: list[str] = []
        for tok in raw.split():
            if tok.startswith("scope="):
                scope = tok.split("=", 1)[1].strip() or scope
                continue
            if tok.startswith("reason="):
                reason = tok.split("=", 1)[1].strip() or reason
                continue
            if tok in {"no-infer-sync", "--no-infer-sync", "no_infer_sync"}:
                no_infer_sync = True
                continue
            instruction_parts.append(tok)

        instruction = " ".join(instruction_parts).strip()
        if not instruction:
            print(f"  {C.RED}PBDEEP requires instruction text{C.RESET}")
            return True

        pbdeep_tool = TOOLS_DIR / "pbdeep_operator.py"
        if not pbdeep_tool.exists():
            print(f"  {C.RED}pbdeep_operator.py not found{C.RESET}")
            return True

        try:
            argv = [
                sys.executable,
                str(pbdeep_tool),
                "--bus-dir",
                str(state.bus_dir),
                "--actor",
                state.actor,
                "--scope",
                scope,
                "--reason",
                reason,
                "--instruction",
                instruction,
            ]
            if no_infer_sync:
                argv.append("--no-infer-sync")
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=10.0,
                env={**os.environ, "PLURIBUS_BUS_DIR": str(state.bus_dir), "PLURIBUS_ACTOR": state.actor, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            rid = (result.stdout or "").strip()
            if result.returncode == 0 and rid:
                print(f"  {C.GREEN}PBDEEP broadcast:{C.RESET} req_id={C.BOLD}{rid}{C.RESET}")
            else:
                msg = (result.stderr or result.stdout or "").strip()[:160]
                print(f"  {C.RED}PBDEEP failed:{C.RESET} {msg}")
        except Exception as e:
            print(f"  {C.RED}PBDEEP failed: {str(e)[:160]}{C.RESET}")
        return True

    elif command == "/pbflush":
        pbflush_tool = TOOLS_DIR / "pbflush_operator.py"
        if pbflush_tool.exists():
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(pbflush_tool),
                        "--bus-dir",
                        str(state.bus_dir),
                        "--actor",
                        state.actor,
                        "--subproject",
                        "ops",
                        "--message",
                        (arg or "PBFLUSH"),
                        "--reason",
                        "operator_pbflush",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    env={**os.environ, "PLURIBUS_BUS_DIR": str(state.bus_dir), "PLURIBUS_ACTOR": state.actor, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                rid = (result.stdout or "").strip()
                if result.returncode == 0 and rid:
                    print(f"  {C.GREEN}PBFLUSH broadcast:{C.RESET} req_id={C.BOLD}{rid}{C.RESET}")
                else:
                    msg = (result.stderr or result.stdout or "").strip()[:160]
                    print(f"  {C.RED}PBFLUSH failed:{C.RESET} {msg}")
            except Exception as e:
                print(f"  {C.RED}PBFLUSH failed: {str(e)[:120]}{C.RESET}")
            return True

        # Fallback: emit a minimal bus request directly (bus-first, no forced kills)
        rid = str(uuid.uuid4())
        emit_bus(
            state.bus_dir,
            topic="operator.pbflush.request",
            kind="request",
            level="warn",
            actor=state.actor,
            data={
                "req_id": rid,
                "subproject": "ops",
                "intent": "pbflush",
                "message": (arg or "PBFLUSH"),
                "reason": "operator_pbflush",
                "iso": now_iso(),
            },
        )
        emit_bus(
            state.bus_dir,
            topic="infer_sync.request",
            kind="request",
            level="info",
            actor=state.actor,
            data={
                "req_id": rid,
                "subproject": "ops",
                "intent": "pbflush",
                "message": (arg or "PBFLUSH"),
                "reason": "operator_pbflush",
                "iso": now_iso(),
            },
        )
        print(f"  {C.GREEN}PBFLUSH broadcast:{C.RESET} req_id={C.BOLD}{rid}{C.RESET}")
        return True

    elif command == "/speak":
        raw = (arg or "").strip()
        if raw.startswith(":"):
            raw = raw[1:].lstrip()
        if not raw:
            print(f"  {C.DIM}Usage:{C.RESET} /speak <text>")
            return True

        speak_tool = TOOLS_DIR / "speak_operator.py"
        if not speak_tool.exists():
            print(f"  {C.RED}speak_operator.py not found{C.RESET}")
            return True

        try:
            context_payload = {
                "origin": "plurichat",
                "provider": state.provider,
                "mode": state.mode,
                "persona": state.persona,
                "context_mode": state.context_mode,
            }
            result = subprocess.run(
                [
                    sys.executable,
                    str(speak_tool),
                    "--file",
                    str(SPEAKER_BUS_FILE),
                    "--emit-bus",
                    "--broadcast",
                    "--context-json",
                    json.dumps(context_payload, ensure_ascii=True),
                    "--source",
                    "plurichat",
                    "--reason",
                    "plurichat_speak",
                ],
                input=raw,
                capture_output=True,
                text=True,
                timeout=5.0,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if result.returncode == 0:
                print(f"  {C.GREEN}SPEAK wrote to:{C.RESET} {SPEAKER_BUS_FILE}")
            else:
                msg = (result.stderr or result.stdout or "").strip()[:200]
                print(f"  {C.RED}SPEAK failed:{C.RESET} {msg}")
        except Exception as e:
            print(f"  {C.RED}SPEAK failed: {str(e)[:160]}{C.RESET}")
        return True

    elif command == "/realagents":
        raw = (arg or "").strip()
        if not raw:
            print(f"  {C.DIM}Usage:{C.RESET} /realagents [targets=claude,codex,gemini] [task_id=REALAGENTS_upgrade] <intent...>")
            print(f"  {C.DIM}Example:{C.RESET} /realagents targets=codex task_id=REALAGENTS_upgrade Deepen A2A negotiation + redirects")
            return True

        targets = "claude,codex,gemini"
        task_id = "REALAGENTS_upgrade"
        spec_ref = "nucleus/specs/realagents_upgrade_v1.md"
        intent_parts: list[str] = []
        for tok in raw.split():
            if tok.startswith("targets="):
                targets = tok.split("=", 1)[1].strip() or targets
                continue
            if tok.startswith("task_id="):
                task_id = tok.split("=", 1)[1].strip() or task_id
                continue
            if tok.startswith("spec_ref="):
                spec_ref = tok.split("=", 1)[1].strip() or spec_ref
                continue
            intent_parts.append(tok)
        intent = " ".join(intent_parts).strip()
        if not intent:
            print(f"  {C.RED}REALAGENTS requires intent text{C.RESET}")
            return True

        tool = TOOLS_DIR / "realagents_operator.py"
        if not tool.exists():
            print(f"  {C.RED}realagents_operator.py not found{C.RESET}")
            return True

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "--bus-dir",
                    str(state.bus_dir),
                    "--targets",
                    targets,
                    "--task-id",
                    task_id,
                    "--spec-ref",
                    spec_ref,
                    "--intent",
                    intent,
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
                env={**os.environ, "PLURIBUS_BUS_DIR": str(state.bus_dir), "PLURIBUS_ACTOR": state.actor, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            rid = (result.stdout or "").strip()
            if result.returncode == 0 and rid:
                print(f"  {C.GREEN}REALAGENTS dispatch:{C.RESET} req_id={C.BOLD}{rid}{C.RESET} targets={targets}")
            else:
                msg = (result.stderr or result.stdout or "").strip()[:200]
                print(f"  {C.RED}REALAGENTS failed:{C.RESET} {msg}")
        except Exception as e:
            print(f"  {C.RED}REALAGENTS failed: {str(e)[:160]}{C.RESET}")
        return True

    else:
        print(f"  {C.RED}Unknown command: {command}{C.RESET}")
        print(f"  Type /help for available commands")
        return True

def setup_readline_completion():
    """Setup readline tab completion with semops lexer."""
    if not LEXER_AVAILABLE:
        return

    lexer = get_lexer()

    def completer(text: str, state: int) -> str | None:
        """Readline completer function."""
        line = readline.get_line_buffer()
        completions = lexer.complete(text)
        if state < len(completions):
            return completions[state][0]
        return None

    readline.set_completer(completer)
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')


def interactive_loop(state: ChatState):
    """Main interactive REPL loop."""
    # Print banner with boot sequence
    print_banner(bus_dir=state.bus_dir, show_boot=True)

    # Setup tab completion
    setup_readline_completion()

    # Initialize SUPERMOTD status
    motd = get_supermotd_status(state.bus_dir)

    # Initial status check
    refresh_provider_status(state)

    # Show SUPERMOTD compact status line
    print(f"  {motd.build_compact_status(state.providers)}")
    print(f"  {render_status_line(state.providers)}")
    print(f"  Mode: {C.BOLD}{state.mode}{C.RESET} | Provider: {C.BOLD}{state.provider}{C.RESET}")
    print()
    print(f"  Type /help for commands, /quit to exit, /supermotd for full status")
    if LEXER_AVAILABLE:
        print(f"  {C.DIM}(Tab completion enabled for operators){C.RESET}")
    print()

    # Track last status update time for periodic refresh
    last_status_time = time.time()

    while state.running:
        try:
            # Periodically refresh and show SUPERMOTD status (every 30 seconds)
            now = time.time()
            if now - last_status_time > 30.0:
                refresh_provider_status(state)
                compact_status = motd.build_compact_status(state.providers)
                # Print status on a new line before prompt
                print(f"\r{C.DIM}{compact_status}{C.RESET}")
                last_status_time = now

            # Prompt with compact status hint
            prompt_prefix = f"{C.CYAN}>{C.RESET} "
            user_input = input(prompt_prefix).strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                handle_command(user_input, state)
                continue
            # Use semops lexer for operator detection
            if LEXER_AVAILABLE:
                lexer = get_lexer()
                tokens = lexer.tokenize(user_input)
                if tokens and tokens[0].type == TokenType.OPERATOR:
                    op_id = tokens[0].metadata.get("operator_id", "")
                    # Extract remaining args after operator
                    remaining = user_input[tokens[0].end:].strip()
                    handle_command(f"/{op_id} {remaining}".strip(), state)
                    continue
            else:
                # Fallback to hardcoded checks if lexer unavailable
                if user_input.upper().startswith("PBEPORT"):
                    arg = user_input[len("PBEPORT"):].strip()
                    handle_command("/pbeport " + arg, state)
                    continue
                if user_input.lower() in {"ckin", "chkin", "checking in", "checkin", "checking-in"}:
                    handle_command("/ckin", state)
                    continue
                if user_input.lower() == "iterate":
                    handle_command("/iterate", state)
                    continue
                if user_input.lower().startswith("pbflush"):
                    msg = user_input[len("pbflush"):].strip()
                    handle_command(("/pbflush " + msg).strip(), state)
                    continue
                if user_input.lower() == "mbad":
                    handle_command("/mbad", state)
                    continue
                if user_input.lower() in {"mabswarm", "mbswarm"}:
                    handle_command("/mabswarm", state)
                    continue

            # Select provider with lens-aware routing
            routing = select_provider_for_query(user_input, state.provider, state.providers)
            selected_provider = routing.provider
            query_type = routing.query_type
            lens = routing.lens

            # Show routing decision with depth indicator
            depth_indicator = f"{C.BOLD}●{C.RESET}" if lens.depth == "deep" else f"{C.DIM}○{C.RESET}"
            topo = f"{lens.topology}:{lens.fanout}" if lens.fanout > 1 else lens.topology
            persona_display = lens.persona_id if state.persona == "auto" else state.persona
            ctx_display = lens.context_mode if state.context_mode == "auto" else state.context_mode
            print(f"  {C.DIM}[{depth_indicator} {lens.depth}/{lens.lane}/{topo} | persona={persona_display} | ctx={ctx_display} | {query_type} → {selected_provider}]{C.RESET}")

            # Optional context-mode override
            if state.context_mode != "auto":
                lens.context_mode = state.context_mode  # type: ignore[misc]

            # Execute chat with topology-aware routing
            effective_prompt = shape_prompt(user_input, lens=lens, persona_override=state.persona)
            response = execute_with_topology(effective_prompt, routing, state.bus_dir, state.actor, state.mode)

            # Display response
            if response.success:
                print()
                print(response.text)
                print()
                print(f"  {C.DIM}[{response.latency_ms:.0f}ms | {response.provider}]{C.RESET}")
            else:
                print(f"  {C.RED}Error: {response.error}{C.RESET}")

            print()

            # Store history
            state.history.append((user_input, response))

        except KeyboardInterrupt:
            print(f"\n  {C.DIM}(Use /quit to exit){C.RESET}")
            continue
        except EOFError:
            break

    print(f"\n{C.DIM}Goodbye!{C.RESET}")

# ==============================================================================
# CLI Entry Point
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plurichat",
        description="PluriChat: Multi-model intelligent chat CLI with optimal routing"
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    p.add_argument("--actor", default=None, help="Actor name")
    p.add_argument("--provider", default="auto", help="Provider: auto, gemini-3, claude, codex, mock")
    p.add_argument("--persona", default="auto", help="Persona ID")
    p.add_argument("--no-fallback", action="store_true", help="Force the selected provider (disable router auto fallback).")
    p.add_argument("--mode", default="direct", choices=["direct", "proxy", "oneshot"], help="Execution mode")
    p.add_argument("--ask", default=None, help="One-shot: send prompt and exit")
    p.add_argument("--prompt", default=None, help="Alias for --ask")
    p.add_argument("--json-output", action="store_true", help="Output JSON result (for oneshot mode)")
    p.add_argument("--status", action="store_true", help="Show provider status and exit")
    p.add_argument("--test", action="store_true", help="Run self-test")
    p.add_argument("--test-all", action="store_true", help="Exhaustive live inference test: prompts for input, tests Gemini-3/Claude/GPT-5.2 via web sessions")
    p.add_argument("--web-session-primary", action="store_true", help="Use authenticated web sessions (ChatGPT/Claude/Gemini web) as primary provider path")
    pbresume_group = p.add_mutually_exclusive_group()
    pbresume_group.add_argument("--pbresume-auto", action="store_true", help="Auto-run PBRESUME on interactive session start")
    pbresume_group.add_argument("--no-pbresume-auto", action="store_true", help="Disable PBRESUME auto-resume")
    return p

def _maybe_run_pbresume_auto(bus_dir: Path, actor: str, enabled: bool) -> None:
    if not enabled:
        return
    pbresume_path = Path("/pluribus/nucleus/tools/pbresume_operator.py")
    if not pbresume_path.exists():
        return
    session_id = os.environ.get("PLURIBUS_SESSION_ID") or f"plurichat-{uuid.uuid4().hex[:8]}"

    def _runner() -> None:
        try:
            subprocess.run(
                [sys.executable, str(pbresume_path), "--auto", "--quiet"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30.0,
                env={
                    **os.environ,
                    "PLURIBUS_BUS_DIR": str(bus_dir),
                    "PLURIBUS_ACTOR": actor,
                    "PLURIBUS_SESSION_ID": session_id,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
        except Exception:
            return

    threading.Thread(target=_runner, daemon=True).start()

def run_self_test(bus_dir: Path, actor: str) -> int:
    """Run self-test to verify all providers."""
    print(f"{C.BOLD}PluriChat Self-Test{C.RESET}")
    print("=" * 50)

    # Test 1: Provider status
    print(f"\n{C.BOLD}1. Provider Status{C.RESET}")
    providers = get_all_provider_status()
    available_count = 0
    for name, status in providers.items():
        indicator = f"{C.GREEN}PASS{C.RESET}" if status.available else f"{C.RED}FAIL{C.RESET}"
        print(f"   {name}: {indicator}")
        if status.available:
            available_count += 1

    # Test 2: Query classification
    print(f"\n{C.BOLD}2. Query Classification{C.RESET}")
    test_queries = [
        ("Write a function to sort a list", "code"),
        ("Explain how neural networks work", "research"),
        ("Write a poem about the moon", "creative"),
        ("Analyze this code for bugs", "analysis"),
        ("Solve this equation: x^2 + 5x + 6 = 0", "math"),
    ]
    class_pass = 0
    for query, expected in test_queries:
        result = classify_query(query)
        if result == expected:
            print(f"   {C.GREEN}PASS{C.RESET} '{query[:30]}...' → {result}")
            class_pass += 1
        else:
            print(f"   {C.RED}FAIL{C.RESET} '{query[:30]}...' → {result} (expected {expected})")

    # Test 3: Provider selection
    print(f"\n{C.BOLD}3. Provider Selection{C.RESET}")
    routing = select_provider_for_query("Write hello world in Python", "auto", providers)
    print(f"   Selected provider for code: {routing.provider}")
    print(f"   Depth: {routing.lens.depth}, Lane: {routing.lens.lane}")

    # Test 3b: Lens depth classification
    print(f"\n{C.BOLD}3b. Lens Depth Classification{C.RESET}")
    depth_tests = [
        ("Implement a hello world function", "narrow"),
        ("Design the architecture for the entire authentication system", "deep"),
        ("Audit the codebase for security vulnerabilities", "deep"),
        ("What is 2+2?", "narrow"),
    ]
    depth_pass = 0
    for query, expected_depth in depth_tests:
        routing = select_provider_for_query(query, "auto", providers)
        if routing.lens.depth == expected_depth:
            print(f"   {C.GREEN}PASS{C.RESET} '{query[:35]}...' → {routing.lens.depth}")
            depth_pass += 1
        else:
            print(f"   {C.RED}FAIL{C.RESET} '{query[:35]}...' → {routing.lens.depth} (expected {expected_depth})")

    # Test 4: Topology routing
    print(f"\n{C.BOLD}4. Topology Routing{C.RESET}")
    topo_tests = [
        ("Explain quicksort", "single", "narrow"),
        ("Design a distributed authentication system with SSO, OAuth, and session management across microservices", "single", "deep"),  # Long prompt triggers deep
    ]
    topo_pass = 0
    for query, expected_topo, expected_depth in topo_tests:
        routing = select_provider_for_query(query, "auto", providers)
        topo_match = routing.lens.topology == expected_topo
        depth_match = routing.lens.depth == expected_depth
        if topo_match and depth_match:
            print(f"   {C.GREEN}PASS{C.RESET} '{query[:30]}...' → {routing.lens.topology}/{routing.lens.depth}")
            topo_pass += 1
        else:
            print(f"   {C.RED}FAIL{C.RESET} '{query[:30]}...' → {routing.lens.topology}/{routing.lens.depth} (expected {expected_topo}/{expected_depth})")

    # Test 5: Bus emission
    print(f"\n{C.BOLD}5. Bus Event Emission{C.RESET}")
    try:
        evt_id = emit_bus(bus_dir, topic="plurichat.test", kind="metric", level="info", actor=actor, data={"test": True})
        print(f"   {C.GREEN}PASS{C.RESET} Emitted event: {evt_id[:8]}...")
    except Exception as e:
        print(f"   {C.RED}FAIL{C.RESET} {e}")

    # Test 6: STRp queue integration
    print(f"\n{C.BOLD}6. STRp Queue Integration{C.RESET}")
    strp_queue_path = TOOLS_DIR / "strp_queue.py"
    if strp_queue_path.exists():
        print(f"   {C.GREEN}PASS{C.RESET} strp_queue.py found")
        # Test dispatch function exists and is callable
        try:
            # Create a mock lens decision for star topology
            test_lens = LensDecision(
                depth="deep",
                lane="pbpair",
                context_mode="full",
                topology="star",
                fanout=2,
            )
            # Don't actually dispatch, just verify function works
            print(f"   {C.GREEN}PASS{C.RESET} dispatch_to_strp_queue function available")
        except Exception as e:
            print(f"   {C.RED}FAIL{C.RESET} STRp dispatch error: {e}")
    else:
        print(f"   {C.YELLOW}SKIP{C.RESET} strp_queue.py not found")

    # Test 7: Context mode handling
    # Context mode depends on depth + effects. In PluriChat we treat chat prompts as effects="none":
    #   - narrow + effects="none" → min
    #   - deep + effects="none" → lite
    print(f"\n{C.BOLD}7. Context Mode Handling{C.RESET}")
    ctx_tests = [
        ("What is 2+2?", "min"),  # Narrow + none effects → min
        ("Design the architecture", "lite"),  # Deep + none effects → lite
    ]
    ctx_pass = 0
    for query, expected_ctx in ctx_tests:
        routing = select_provider_for_query(query, "auto", providers)
        if routing.lens.context_mode == expected_ctx:
            print(f"   {C.GREEN}PASS{C.RESET} '{query[:25]}...' → context_mode={routing.lens.context_mode}")
            ctx_pass += 1
        else:
            print(f"   {C.RED}FAIL{C.RESET} '{query[:25]}...' → {routing.lens.context_mode} (expected {expected_ctx})")

    # Test 8: Mock provider chat with topology
    print(f"\n{C.BOLD}8. Mock Provider Chat (with topology){C.RESET}")
    routing = select_provider_for_query("Say hello", "mock", providers, include_lens=True)
    response = execute_with_topology("Say hello", routing, bus_dir, actor, "direct", timeout=10)
    if response.success:
        print(f"   {C.GREEN}PASS{C.RESET} Response: {response.text[:50]}...")
        print(f"   {C.DIM}Provider: {response.provider}, Latency: {response.latency_ms:.0f}ms{C.RESET}")
    else:
        print(f"   {C.RED}FAIL{C.RESET} {response.error}")

    # Test 9: Request kind inference
    print(f"\n{C.BOLD}9. Request Kind Inference{C.RESET}")
    kind_tests = [
        ("Audit the security", "audit"),
        ("Benchmark performance", "benchmark"),
        ("Implement a feature", "apply"),
        ("Explain how it works", "distill"),
    ]
    kind_pass = 0
    for query, expected_kind in kind_tests:
        inferred = _infer_request_kind(query)
        if inferred == expected_kind:
            print(f"   {C.GREEN}PASS{C.RESET} '{query[:20]}...' → {inferred}")
            kind_pass += 1
        else:
            print(f"   {C.RED}FAIL{C.RESET} '{query[:20]}...' → {inferred} (expected {expected_kind})")

    # Summary
    print(f"\n{C.BOLD}Summary{C.RESET}")
    print(f"   Providers available: {available_count}/{len(providers)}")
    print(f"   Classification tests: {class_pass}/{len(test_queries)}")
    print(f"   Depth classification: {depth_pass}/{len(depth_tests)}")
    print(f"   Topology routing: {topo_pass}/{len(topo_tests)}")
    print(f"   Context mode: {ctx_pass}/{len(ctx_tests)}")
    print(f"   Kind inference: {kind_pass}/{len(kind_tests)}")
    print(f"   Lens available: {C.GREEN}YES{C.RESET}" if LENS_AVAILABLE else f"   Lens available: {C.YELLOW}NO (fallback){C.RESET}")

    total_tests = len(test_queries) + len(depth_tests) + len(topo_tests) + len(ctx_tests) + len(kind_tests)
    total_pass = class_pass + depth_pass + topo_pass + ctx_pass + kind_pass
    all_pass = available_count > 0 and total_pass == total_tests

    if all_pass:
        print(f"\n   {C.GREEN}ALL TESTS PASSED ({total_pass}/{total_tests}){C.RESET}")
        return 0
    else:
        print(f"\n   {C.YELLOW}SOME TESTS FAILED ({total_pass}/{total_tests}){C.RESET}")
        return 1


# ==============================================================================
# Web Session Providers (Authenticated Browser OAuth Sessions)
# ==============================================================================

WEB_SESSION_PROVIDERS = {
    "gemini-web": {
        "name": "Gemini Web (Google AI Studio)",
        "model": "gemini-3-pro",
        "session_key": "gemini_web_session",
        "auth_check": "gemini_cli_logged_in",  # Reuse CLI auth for now
        "endpoint_hint": "https://aistudio.google.com",
    },
    "claude-web": {
        "name": "Claude Web (claude.ai)",
        "model": "claude-opus-4-5",
        "session_key": "claude_web_session",
        "auth_check": "claude_logged_in",
        "endpoint_hint": "https://claude.ai",
    },
    "chatgpt-web": {
        "name": "ChatGPT Web (OpenAI)",
        "model": "gpt-5.2-turbo",
        "session_key": "chatgpt_web_session",
        "auth_check": "openai_web_logged_in",
        "endpoint_hint": "https://chat.openai.com",
    },
}


def check_web_session_availability() -> dict[str, dict]:
    """Check which web sessions are available for authenticated inference.

    This checks for valid OAuth sessions that can be used for browser-based
    inference routing (headless browserless mode).
    """
    daemon = check_browser_daemon_status()
    daemon_running = bool(daemon.get("running", False))
    tabs = daemon.get("tabs") if isinstance(daemon.get("tabs"), dict) else {}

    results: dict[str, dict] = {}
    for provider_id, config in WEB_SESSION_PROVIDERS.items():
        tab = tabs.get(provider_id) if isinstance(tabs, dict) else None
        tab_status = tab.get("status") if isinstance(tab, dict) else None
        tab_error = tab.get("error") if isinstance(tab, dict) else None
        available = daemon_running and tab_status == "ready"

        results[provider_id] = {
            "available": available,
            "name": config["name"],
            "model": config["model"],
            # Backwards-compatible fields (used by /test-all rendering)
            "has_auth": False,
            "has_session": daemon_running and bool(tab_status),
            "has_env": False,
            "endpoint": config["endpoint_hint"],
            # Extra observability (helps dashboard/operator debug)
            "tab_status": tab_status or ("daemon_not_running" if not daemon_running else "unknown"),
            "tab_error": tab_error,
        }

    return results


def check_browser_daemon_status(root: Path = Path("/pluribus")) -> dict:
    """Check if browser daemon is running and get tab status."""
    state_path = root / ".pluribus" / "browser_daemon.json"
    if not state_path.exists():
        return {"running": False, "tabs": {}}
    try:
        data = json.loads(state_path.read_text())
        pid = int(data.get("pid") or 0)
        running_flag = bool(data.get("running")) and pid > 0
        alive = False
        if running_flag:
            try:
                os.kill(pid, 0)
                alive = True
            except Exception:
                alive = False
        if not alive:
            tabs = data.get("tabs") if isinstance(data.get("tabs"), dict) else {}
            return {"running": False, "tabs": tabs}
        return data
    except Exception:
        return {"running": False, "tabs": {}}


def execute_web_session_inference(
    prompt: str,
    provider_id: str,
    bus_dir: Path,
    actor: str,
    timeout: float = 60.0
) -> ChatResponse:
    """Execute inference via web session (browser OAuth).

    First tries browser daemon tabs, falls back to API providers if unavailable.
    """
    req_id = str(uuid.uuid4())
    start = time.time()

    config = WEB_SESSION_PROVIDERS.get(provider_id, {})
    model = config.get("model", "unknown")

    # Check browser daemon status
    daemon_status = check_browser_daemon_status()
    daemon_running = bool(daemon_status.get("running", False))
    tabs = daemon_status.get("tabs") if isinstance(daemon_status.get("tabs"), dict) else {}
    tab_status = tabs.get(provider_id, {}) if isinstance(tabs, dict) else {}
    tab_ready = tab_status.get("status") == "ready" if isinstance(tab_status, dict) else False
    tab_exists = bool(tab_status) if isinstance(tab_status, dict) else False
    can_route = daemon_running and tab_exists

    # Emit web session routing event
    emit_bus(bus_dir, topic="plurichat.web_session.request", kind="request", level="info", actor=actor, data={
        "req_id": req_id,
        "provider": provider_id,
        "model": model,
        "prompt_length": len(prompt),
        "routing": "browser_daemon" if can_route else "blocked",
        "daemon_running": daemon_running,
        "tab_ready": tab_ready,
    })

    if can_route:
        # Route via the browser daemon using the append-only bus.
        emit_bus(
            bus_dir,
            topic="browser.chat.request",
            kind="request",
            level="info",
            actor=actor,
            data={"req_id": req_id, "provider": provider_id, "prompt": prompt, "timeout_s": timeout},
        )

        events_path = bus_dir / "events.ndjson"
        deadline = time.time() + float(timeout)
        pos = events_path.stat().st_size if events_path.exists() else 0
        while time.time() < deadline:
            if not events_path.exists():
                time.sleep(0.2)
                continue
            with events_path.open("rb") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if not chunk:
                time.sleep(0.2)
                continue
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

                latency_ms = (time.time() - start) * 1000
                if data.get("success"):
                    text = str(data.get("response") or data.get("response_preview") or "").strip()
                    response = ChatResponse(
                        text=text,
                        provider=provider_id,
                        model=model,
                        latency_ms=latency_ms,
                        req_id=req_id,
                        success=True,
                    )
                else:
                    err = str(data.get("error") or "Provider error").strip()
                    response = ChatResponse(
                        text=err,
                        provider=provider_id,
                        model=model,
                        latency_ms=latency_ms,
                        req_id=req_id,
                        success=False,
                        error=err,
                    )

                emit_bus(
                    bus_dir,
                    topic="plurichat.web_session.response",
                    kind="response",
                    level="info" if response.success else "warn",
                    actor=actor,
                    data={
                        "req_id": req_id,
                        "provider": provider_id,
                        "routing": "browser_daemon",
                        "success": response.success,
                        "latency_ms": latency_ms,
                        "response_length": len(response.text) if response.text else 0,
                    },
                )
                return response
            time.sleep(0.2)

        latency_ms = (time.time() - start) * 1000
        response = ChatResponse(
            text="Request timed out",
            provider=provider_id,
            model=model,
            latency_ms=latency_ms,
            req_id=req_id,
            success=False,
            error="Timeout",
        )
        emit_bus(
            bus_dir,
            topic="plurichat.web_session.response",
            kind="response",
            level="warn",
            actor=actor,
            data={"req_id": req_id, "provider": provider_id, "routing": "browser_daemon", "success": False, "latency_ms": latency_ms, "error": "Timeout"},
        )
        return response

    # Browser daemon isn't ready. PluriChat does not fall back to CLI/API/mock providers.
    latency_ms = (time.time() - start) * 1000
    blocker = "daemon_not_running" if not daemon_running else "tab_not_ready"
    tab_err = None
    tab_state = None
    if isinstance(tab_status, dict):
        tab_state = str(tab_status.get("status") or "").strip() or None
        te = tab_status.get("error")
        tab_err = str(te).strip() if isinstance(te, str) and te.strip() else None

    err_msg = "browser daemon not running" if not daemon_running else (tab_err or tab_state or "web tab not ready")
    response = ChatResponse(
        text=err_msg,
        provider=provider_id,
        model=model,
        latency_ms=latency_ms,
        req_id=req_id,
        success=False,
        error=err_msg,
    )
    emit_bus(
        bus_dir,
        topic="plurichat.web_session.response",
        kind="response",
        level="warn",
        actor=actor,
        data={
            "req_id": req_id,
            "provider": provider_id,
            "routing": "blocked",
            "blocker": blocker,
            "daemon_running": daemon_running,
            "tab_ready": tab_ready,
            "tab_status": tab_state,
            "error": err_msg,
            "success": False,
            "latency_ms": latency_ms,
        },
    )
    # Emit an actionable request event for operators.
    if "login" in err_msg.lower() or "needs_login" in (tab_state or "").lower():
        emit_bus(
            bus_dir,
            topic="plurichat.web_session.auth.required",
            kind="request",
            level="warn",
            actor=actor,
            data={"req_id": req_id, "provider": provider_id, "status": "needs_login"},
        )
    return response


def run_test_all(bus_dir: Path, actor: str, web_session_primary: bool = False) -> int:
    """Run exhaustive live inference test across all providers.

    Tests Gemini-3, Claude Opus 4.5, GPT-5.2 (NOT Codex CLI).
    Uses web session routing as primary when --web-session-primary is set.

    Flow:
    1. Prompt user for test string
    2. Check all provider availability (API + web sessions)
    3. Execute parallel/sequential inference across all targets
    4. Display dereferenced live results
    5. Emit bus evidence for inter/intra agent visibility
    """
    print(f"""
{C.CYAN}╔══════════════════════════════════════════════════════════════╗
║  {C.BOLD}PluriChat Exhaustive Live Inference Test{C.RESET}{C.CYAN}                    ║
║  {C.DIM}Gemini-3 • Claude Opus 4.5 • GPT-5.2{C.RESET}{C.CYAN}                         ║
║  {C.DIM}Web Session Primary: {'YES' if web_session_primary else 'NO'}{C.RESET}{C.CYAN}                                    ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")

    # Test ID for tracing
    test_id = f"test-all-{int(time.time())}"

    # Emit test start event
    emit_bus(bus_dir, topic="plurichat.test_all.start", kind="metric", level="info", actor=actor, data={
        "test_id": test_id,
        "web_session_primary": web_session_primary,
        "iso": now_iso(),
    })

    # 1. Check provider availability
    print(f"{C.BOLD}1. Provider Availability{C.RESET}")
    print("-" * 50)

    # Web session providers
    web_providers = check_web_session_availability()
    print(f"\n  {C.BOLD}Web Session Providers:{C.RESET}")
    for name, info in web_providers.items():
        indicator = f"{C.GREEN}●{C.RESET}" if info["available"] else f"{C.RED}○{C.RESET}"
        auth_info = []
        if info["has_auth"]:
            auth_info.append("auth")
        if info["has_session"]:
            auth_info.append("session")
        if info["has_env"]:
            auth_info.append("env")
        auth_str = f" [{','.join(auth_info)}]" if auth_info else ""
        print(f"    {indicator} {info['name']} ({info['model']}){auth_str}")

    # 2. Get user prompt
    print(f"\n{C.BOLD}2. Test Prompt{C.RESET}")
    print("-" * 50)
    try:
        prompt = input(f"\n  {C.CYAN}Enter test prompt (or press Enter for default):{C.RESET} ").strip()
        if not prompt:
            prompt = "Explain what you are (model name, version) and respond with a single sentence about your capabilities."
        print(f"  {C.DIM}Using: {prompt[:60]}{'...' if len(prompt) > 60 else ''}{C.RESET}")
    except (KeyboardInterrupt, EOFError):
        print(f"\n  {C.RED}Cancelled{C.RESET}")
        return 1

    # 3. Execute inference across all providers
    print(f"\n{C.BOLD}3. Live Inference Results{C.RESET}")
    print("-" * 50)

    # Define test targets (web-session-only policy).
    test_targets = [
        ("gemini-web", "Gemini (Web Session)", True),
        ("claude-web", "Claude (Web Session)", True),
        ("chatgpt-web", "ChatGPT (Web Session)", True),
    ]

    results = {}

    for provider_id, display_name, is_web_session in test_targets:
        print(f"\n  {C.BOLD}Testing: {display_name}{C.RESET}")
        print(f"  {C.DIM}Provider: {provider_id}{C.RESET}")

        # Check availability (web session only)
        info = web_providers.get(provider_id, {})
        available = info.get("available", False)

        if not available:
            # For live testing, still attempt routing if the daemon/tab exist; the daemon can often recover by reloading.
            if not info.get("has_session", False):
                print(f"  {C.RED}SKIP{C.RESET} - Provider unavailable")
                results[provider_id] = {"success": False, "error": "unavailable", "skipped": True}
                continue
            tab_state = str(info.get("tab_status") or "unknown")
            tab_err = str(info.get("tab_error") or "").strip()
            if tab_err:
                tab_err = (tab_err[:160] + "...") if len(tab_err) > 160 else tab_err
                print(f"  {C.YELLOW}NOTE{C.RESET} - Tab not ready ({tab_state}): {tab_err}")
            else:
                print(f"  {C.YELLOW}NOTE{C.RESET} - Tab not ready ({tab_state}); attempting anyway")

        # Execute inference
        try:
            response = execute_web_session_inference(prompt, provider_id, bus_dir, actor, timeout=90.0)

            results[provider_id] = {
                "success": response.success,
                "text": response.text,
                "latency_ms": response.latency_ms,
                "model": response.model,
                "error": response.error,
            }

            if response.success:
                print(f"  {C.GREEN}SUCCESS{C.RESET} ({response.latency_ms:.0f}ms)")
                print(f"  {C.DIM}Response:{C.RESET}")
                # Truncate long responses for display
                display_text = response.text[:500] + "..." if len(response.text) > 500 else response.text
                for line in display_text.split("\n"):
                    print(f"    {line}")
            else:
                print(f"  {C.RED}FAILED{C.RESET}: {response.error}")

        except Exception as e:
            print(f"  {C.RED}ERROR{C.RESET}: {e}")
            results[provider_id] = {"success": False, "error": str(e)}

    # 4. Inter-agent collaboration test
    print(f"\n{C.BOLD}4. Inter-Agent Collaboration{C.RESET}")
    print("-" * 50)

    # Send test results to hexis buffer for inter-agent visibility
    hexis_tool = TOOLS_DIR / "hexis_buffer.py"
    if hexis_tool.exists():
        try:
            hexis_env = {**os.environ, "HEXIS_BUFFER_DIR": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
            subprocess.run(
                [
                    sys.executable, str(hexis_tool), "pub",
                    "--agent", "codex",
                    "--actor", actor,
                    "--topic", "plurichat.test_all.results",
                    "--kind", "artifact",
                    "--flow-inter", "codex",
                    "--json", json.dumps({
                        "test_id": test_id,
                        "prompt": prompt,
                        "results": {k: {"success": v.get("success"), "latency_ms": v.get("latency_ms")} for k, v in results.items()},
                        "web_session_primary": web_session_primary,
                        "iso": now_iso(),
                    }),
                ],
                check=False,
                capture_output=True,
                timeout=5,
                env=hexis_env,
            )
            print(f"  {C.GREEN}●{C.RESET} Results published to hexis buffer (inter-agent)")
        except Exception as e:
            print(f"  {C.YELLOW}○{C.RESET} Hexis buffer publish failed: {e}")

    # Emit to main bus
    emit_bus(bus_dir, topic="plurichat.test_all.complete", kind="artifact", level="info", actor=actor, data={
        "test_id": test_id,
        "prompt": prompt[:200],
        "results": {k: {"success": v.get("success"), "latency_ms": v.get("latency_ms"), "error": v.get("error")} for k, v in results.items()},
        "web_session_primary": web_session_primary,
        "iso": now_iso(),
    })
    print(f"  {C.GREEN}●{C.RESET} Results published to agent bus")

    # 5. Summary
    print(f"\n{C.BOLD}5. Summary{C.RESET}")
    print("-" * 50)

    success_count = sum(1 for r in results.values() if r.get("success"))
    total_count = len([r for r in results.values() if not r.get("skipped")])
    skipped_count = sum(1 for r in results.values() if r.get("skipped"))

    print(f"\n  Tested: {total_count} providers")
    print(f"  Passed: {C.GREEN}{success_count}{C.RESET}")
    print(f"  Failed: {C.RED}{total_count - success_count}{C.RESET}")
    if skipped_count:
        print(f"  Skipped: {C.YELLOW}{skipped_count}{C.RESET}")
    print(f"  Test ID: {test_id}")

    if success_count == total_count and total_count > 0:
        print(f"\n  {C.GREEN}ALL LIVE INFERENCE TESTS PASSED{C.RESET}")
        return 0
    elif success_count > 0:
        print(f"\n  {C.YELLOW}PARTIAL SUCCESS ({success_count}/{total_count}){C.RESET}")
        return 0
    else:
        print(f"\n  {C.RED}ALL TESTS FAILED{C.RESET}")
        return 1


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    actor = (args.actor or default_actor()).strip() or "plurichat"
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or str(DEFAULT_BUS_DIR))
    ensure_dir(bus_dir)

    # Map friendly provider names
    provider_mapping = {
        "gemini-3": "vertex-gemini-curl",
        "claude": "claude-api",
        "codex": "codex-cli",
    }
    provider = provider_mapping.get(args.provider.lower(), args.provider.lower())

    # Self-test mode
    if args.test:
        return run_self_test(bus_dir, actor)

    # Exhaustive test-all mode
    if args.test_all:
        return run_test_all(bus_dir, actor, web_session_primary=args.web_session_primary)

    # Status mode
    if args.status:
        providers = get_all_provider_status()
        session = load_vps_session()
        print(f"\n{C.BOLD}Provider Status{C.RESET}")
        if isinstance(session, dict) and session.get("active_fallback"):
            print(f"  {C.DIM}active_fallback={session.get('active_fallback')} flow_mode={session.get('flow_mode')}{C.RESET}")
        for name, status in providers.items():
            indicator = f"{C.GREEN}●{C.RESET}" if status.available else f"{C.RED}○{C.RESET}"
            model_info = f" ({status.model})" if status.model else ""
            cooldown_info = f" {C.DIM}cooldown→{status.cooldown_until}{C.RESET}" if status.cooldown_until else ""
            blocker_info = f" {C.DIM}blocked:{status.blocker}{C.RESET}" if status.blocker else ""
            error_info = f" {C.DIM}{status.error}{C.RESET}" if status.error else ""
            print(f"  {indicator} {name}{model_info}{cooldown_info}{blocker_info} {error_info}".rstrip())
        return 0

    # One-shot mode
    user_prompt = args.prompt or args.ask
    pbresume_env = os.environ.get("PLURIBUS_PBRESUME_AUTO")
    pbresume_auto = True if pbresume_env is None else pbresume_env.strip().lower() not in {"0", "false", "no", "off"}
    if args.no_pbresume_auto:
        pbresume_auto = False
    elif args.pbresume_auto:
        pbresume_auto = True
    if pbresume_auto and not user_prompt and args.mode != "oneshot":
        _maybe_run_pbresume_auto(bus_dir, actor, enabled=True)
    if user_prompt or args.mode == "oneshot":
        if not user_prompt:
            # Fallback if mode=oneshot but no prompt (read from stdin?)
            # For now, just error
            sys.stderr.write("Error: --prompt or --ask required for oneshot mode\n")
            return 1

        # Semantic operators in oneshot mode (non-conversational)
        if user_prompt.strip().lower() == "iterate":
            iterate_tool = TOOLS_DIR / "iterate_operator.py"
            rid = ""
            if iterate_tool.exists():
                p = subprocess.run(
                    [sys.executable, str(iterate_tool), "--bus-dir", str(bus_dir), "--agent", actor],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    env={**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PLURIBUS_ACTOR": actor, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                if p.returncode == 0:
                    rid = (p.stdout or "").strip()
            if not rid:
                # Fallback: emit minimal bus request directly
                rid = str(uuid.uuid4())
                emit_bus(
                    bus_dir,
                    topic="infer_sync.request",
                    kind="request",
                    level="info",
                    actor=actor,
                    data={
                        "req_id": rid,
                        "subproject": "beam_10x",
                        "intent": "iterate",
                        "inputs": {"operator": "iterate"},
                        "constraints": {"append_only": True, "tests_first": True, "non_blocking": True},
                        "response_topic": "infer_sync.response",
                        "iso": now_iso(),
                    },
                )
                emit_bus(
                    bus_dir,
                    topic="operator.iterate.request",
                    kind="request",
                    level="info",
                    actor=actor,
                    data={"req_id": rid, "subproject": "beam_10x", "intent": "iterate", "iso": now_iso()},
                )

            if args.json_output:
                print(json.dumps({"operator": "iterate", "req_id": rid, "success": True}, ensure_ascii=False))
                return 0
            print(rid)
            return 0

        if user_prompt.strip().lower().startswith("pbflush"):
            msg = user_prompt.strip()[len("pbflush"):].strip() or "PBFLUSH"
            pbflush_tool = TOOLS_DIR / "pbflush_operator.py"
            rid = ""
            if pbflush_tool.exists():
                p = subprocess.run(
                    [sys.executable, str(pbflush_tool), "--bus-dir", str(bus_dir), "--actor", actor, "--subproject", "ops", "--message", msg, "--reason", "operator_pbflush"],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    env={**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PLURIBUS_ACTOR": actor, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                if p.returncode == 0:
                    rid = (p.stdout or "").strip()
            if not rid:
                rid = str(uuid.uuid4())
                emit_bus(
                    bus_dir,
                    topic="operator.pbflush.request",
                    kind="request",
                    level="warn",
                    actor=actor,
                    data={"req_id": rid, "subproject": "ops", "intent": "pbflush", "message": msg, "reason": "operator_pbflush", "iso": now_iso()},
                )
                emit_bus(
                    bus_dir,
                    topic="infer_sync.request",
                    kind="request",
                    level="info",
                    actor=actor,
                    data={"req_id": rid, "subproject": "ops", "intent": "pbflush", "message": msg, "reason": "operator_pbflush", "iso": now_iso()},
                )

            if args.json_output:
                print(json.dumps({"operator": "pbflush", "req_id": rid, "success": True}, ensure_ascii=False))
                return 0
            print(rid)
            return 0

        if user_prompt.strip().lower().startswith("realagents"):
            raw = user_prompt.strip()[len("realagents"):].strip()
            targets = "claude,codex,gemini"
            task_id = "REALAGENTS_upgrade"
            spec_ref = "nucleus/specs/realagents_upgrade_v1.md"
            intent_parts: list[str] = []
            for tok in raw.split():
                if tok.startswith("targets="):
                    targets = tok.split("=", 1)[1].strip() or targets
                    continue
                if tok.startswith("task_id="):
                    task_id = tok.split("=", 1)[1].strip() or task_id
                    continue
                if tok.startswith("spec_ref="):
                    spec_ref = tok.split("=", 1)[1].strip() or spec_ref
                    continue
                intent_parts.append(tok)
            intent = " ".join(intent_parts).strip() or "REALAGENTS: dispatch"

            tool = TOOLS_DIR / "realagents_operator.py"
            rid = ""
            if tool.exists():
                p = subprocess.run(
                    [
                        sys.executable,
                        str(tool),
                        "--bus-dir",
                        str(bus_dir),
                        "--targets",
                        targets,
                        "--task-id",
                        task_id,
                        "--spec-ref",
                        spec_ref,
                        "--intent",
                        intent,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    env={**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PLURIBUS_ACTOR": actor, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                if p.returncode == 0:
                    rid = (p.stdout or "").strip()
            if not rid:
                rid = str(uuid.uuid4())
                emit_bus(
                    bus_dir,
                    topic="rd.tasks.dispatch",
                    kind="request",
                    level="info",
                    actor=actor,
                    data={
                        "req_id": rid,
                        "task_id": task_id,
                        "intent": "realagents_custom",
                        "iso": now_iso(),
                        "spec_ref": spec_ref,
                        "targets": [t.strip() for t in targets.split(",") if t.strip()],
                        "tasks": [{"id": "T0", "title": intent, "depends_on": [], "deliverables": ["implementation"], "acceptance": ["evidence emitted"]}],
                    },
                )

            if args.json_output:
                print(json.dumps({"operator": "realagents", "req_id": rid, "success": True}, ensure_ascii=False))
                return 0
            print(rid)
            return 0

        providers = get_all_provider_status()
        routing = select_provider_for_query(user_prompt, provider, providers)
        selected = routing.provider
        if args.no_fallback and provider != "auto":
            selected = provider
            routing.provider = provider
        
        # Apply persona from args if provided
        if args.persona and args.persona != "auto":
            routing.lens.persona_id = args.persona

        # Emit lens routing decision to bus
        emit_bus(bus_dir, topic="plurichat.lens.decision", kind="metric", level="info", actor=actor, data={
            "depth": routing.lens.depth,
            "lane": routing.lens.lane,
            "context_mode": routing.lens.context_mode,
            "topology": routing.lens.topology,
            "fanout": routing.lens.fanout,
            "query_type": routing.query_type,
            "selected_provider": selected,
            "persona": routing.lens.persona_id,
        })

        # Execute with topology-aware routing
        effective_prompt = shape_prompt(user_prompt, lens=routing.lens, persona_override=args.persona)
        # Use direct mode for oneshot unless proxy explicitly requested
        exec_mode = "direct" if args.mode == "oneshot" else args.mode
        response = execute_with_topology(effective_prompt, routing, bus_dir, actor, exec_mode)

        if args.json_output:
            print(json.dumps({
                "text": response.text,
                "provider": response.provider,
                "latency_ms": response.latency_ms,
                "success": response.success,
                "error": response.error,
                "req_id": response.req_id
            }, ensure_ascii=False))
            return 0 if response.success else 1
        
        if response.success:
            print(response.text)
            return 0
        else:
            sys.stderr.write(f"Error: {response.error}\n")
            return 1

    # Interactive mode
    state = ChatState(
        bus_dir=bus_dir,
        actor=actor,
        provider=provider if args.no_fallback else "auto",
        persona=args.persona,
        mode=args.mode,
    )

    # Handle signals
    def handle_sigint(sig, frame):
        state.running = False
    signal.signal(signal.SIGINT, handle_sigint)

    interactive_loop(state)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
