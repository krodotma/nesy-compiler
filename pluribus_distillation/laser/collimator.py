#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/pluribus/nucleus/tools")

from topology import select_topology  # noqa: E402
from persona_registry import choose_persona, load_personas  # noqa: E402

# CAGENT v27 integration
def load_cagent_paths(paths_file: Path | None = None) -> dict:
    """Load CAGENT canonical paths registry for citizen compliance."""
    if paths_file is None:
        paths_file = Path('/pluribus/nucleus/specs/cagent_paths.json')
    if not paths_file.exists():
        return {}
    with paths_file.open('r') as f:
        return json.load(f)

def check_cagent_compliance(actor: str, cagent_paths: dict) -> dict:
    """Verify agent meets CAGENT citizenship requirements."""
    compliance = {
        'actor': actor,
        'citizen': True,
        'paths_valid': bool(cagent_paths),
        'constitution_exists': Path('/pluribus/nucleus/specs/CITIZEN.md').exists(),
        'tier': 'SAGENT' if os.environ.get('PLURIBUS_CITIZEN_CLASS') == 'sagent' else 'SWAGENT'
    }
    return compliance



def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: Path | None, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    if not bus_dir:
        return
    evt = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)


def load_vps_session(root: Path) -> dict[str, Any]:
    path = root / ".pluribus" / "vps_session.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def find_pluribus_root(start: Path) -> Path:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    if (Path("/pluribus") / ".pluribus" / "rhizome.json").exists():
        return Path("/pluribus")
    return cur


@dataclass(frozen=True)
class LASERRequest:
    req_id: str
    goal: str
    kind: str
    effects: str
    prefer_providers: list[str]
    require_model_prefix: str | None
    synthesis_mode: str | None = None  # None|lens_laser
    synthesis_config: dict | None = None  # LENS/LASER config if requested


@dataclass(frozen=True)
class RoutePlan:
    req_id: str
    depth: str  # narrow|deep
    lane: str  # dialogos|pbpair
    provider: str
    context_mode: str  # min|lite|full
    topology: str  # single|star|peer_debate
    fanout: int
    topology_reason: str
    persona_id: str
    persona_reason: str
    domains: list[str]  # Extracted domain tags
    target_app: str  # Target app from MANIFEST.yaml
    notes: list[str]
    synthesis_mode: str | None = None  # None|lens_laser
    lens_laser_config: dict | None = None  # Config for LENS/LASER if synthesis_mode == "lens_laser"


def _normalize_kind(v: str) -> str:
    v = (v or "").strip().lower()
    if v in {"distill", "apply", "verify", "audit", "benchmark"}:
        return v
    return "other"


def _normalize_effects(v: str) -> str:
    v = (v or "").strip().lower()
    if v in {"none", "file", "network", "unknown"}:
        return v
    return "unknown"


def _load_manifest(root: Path) -> dict[str, Any]:
    """Load MANIFEST.yaml for app-of-apps routing."""
    manifest_path = root / "MANIFEST.yaml"
    if not manifest_path.exists():
        return {}
    if yaml is None:
        # Fallback: try to parse as simple key-value (won't work for nested)
        return {}
    try:
        return yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _extract_domains(goal: str, manifest: dict[str, Any]) -> list[str]:
    """Extract domain tags from NL prompt using MANIFEST keyword mapping."""
    g = (goal or "").strip().lower()
    domains: list[str] = []
    
    # Get keyword→domain mapping from MANIFEST
    keyword_domains = manifest.get("keyword_domains", {})
    
    # Default keywords if MANIFEST not loaded
    if not keyword_domains:
        keyword_domains = {
            "dashboard": "kroma", "frontend": "kroma", "ui": "kroma", "button": "kroma",
            "bus": "pluribus", "operator": "pluribus", "agent": "pluribus", "protocol": "pluribus",
            "video": "cinema", "avatar": "avtr", "ring": "pqc", "crypto": "pqc",
            "inference": "vision", "ocr": "ocr", "screenshot": "vlm",
            "knowledge": "kg", "graph": "kg", "embedding": "rag", "vector": "rag",
        }
    
    # Match keywords in goal
    for keyword, domain in keyword_domains.items():
        if keyword.lower() in g:
            if domain not in domains:
                domains.append(domain)
    
    # Default to pluribus if no matches
    if not domains:
        domains = ["pluribus"]
    
    return domains


def _route_to_app(domains: list[str], manifest: dict[str, Any]) -> str:
    """Route domains to target app using MANIFEST domain_routing."""
    domain_routing = manifest.get("domain_routing", {})
    
    # Default routing if MANIFEST not loaded
    if not domain_routing:
        domain_routing = {
            "pluribus": "pluribus-core", "omega": "pluribus-core", "pqc": "pluribus-core",
            "kroma": "pluribus-dashboard",
            "cinema": "pluribus-art", "avtr": "pluribus-art",
        }
    
    # Find first matching domain's app
    for domain in domains:
        if domain in domain_routing:
            return domain_routing[domain]
    
    return "pluribus-core"  # Default fallback


def _classify_depth(goal: str, kind: str) -> str:
    g = (goal or "").strip().lower()
    if kind in {"audit", "benchmark"}:
        return "deep"
    if any(w in g for w in ["architecture", "design", "spec", "research", "theory", "protocol", "schema", "dsl", "neurosymbolic", "collimator", "lens"]):
        return "deep"
    if len(g) > 240:
        return "deep"
    return "narrow"


def _pick_lane(depth: str, effects: str) -> tuple[str, str]:
    # Default: Dialogos gives streaming UX and easy bus provenance.
    # PBPAIR lane is used when depth is high and we want structured outputs.
    if depth == "deep":
        return "pbpair", "full" if effects != "none" else "lite"
    return "dialogos", "min" if effects == "none" else "lite"

def _select_topology(depth: str, kind: str, effects: str) -> tuple[str, int, str]:
    # Reuse STRp topology policy (single/star/peer_debate) to express
    # “deep project understanding vs isolated subtask” allocation.
    #
    # Deterministic synthesis:
    # - deep + low tool_density ⇒ allow small fanout
    # - tool-dense (network/unknown effects) ⇒ single (reduce coordination error)
    tool_density = 0.5
    if effects == "none":
        tool_density = 0.2
    elif effects == "file":
        tool_density = 0.5
    elif effects in {"network", "unknown"}:
        tool_density = 0.85

    parallelizable = bool(depth == "deep" and kind in {"distill", "hypothesize", "apply"} and effects != "network")
    coord_budget_tokens = 6500 if depth == "deep" else 1000

    topo = select_topology(
        {
            "kind": kind,
            "parallelizable": parallelizable,
            "tool_density": tool_density,
            "coord_budget_tokens": coord_budget_tokens,
            "topology_hint": "auto",
        }
    )
    return str(topo.get("topology") or "single"), int(topo.get("fanout") or 1), str(topo.get("reason") or "unknown")


def _provider_allowed_by_session(provider: str, session: dict[str, Any]) -> tuple[bool, str | None]:
    # Minimal guardrails so we don't repeatedly choose an unavailable lane.
    provider = (provider or "").strip().lower()
    if provider in {"auto", "mock"}:
        return True, None
    if provider == "claude-cli":
        if session.get("claude_logged_in") is False:
            return False, "claude-cli not logged in"
    if provider == "gemini-cli":
        if session.get("gemini_cli_logged_in") is False:
            return False, "gemini-cli not logged in"
    cooldowns = session.get("provider_cooldowns")
    if isinstance(cooldowns, dict):
        # vps_session uses provider names like 'vertex-curl' etc; this is best-effort.
        # If cooldown exists for either exact key or mapped key, avoid.
        now = time.time()
        v = cooldowns.get(provider)
        if isinstance(v, (int, float)) and now < float(v):
            return False, "provider in cooldown"
    return True, None


def route_query(req: LASERRequest, *, session: dict[str, Any]) -> RoutePlan:
    kind = _normalize_kind(req.kind)
    effects = _normalize_effects(req.effects)
    depth = _classify_depth(req.goal, kind)
    lane, context_mode = _pick_lane(depth, effects)
    topology, fanout, topology_reason = _select_topology(depth, kind, effects)

    notes: list[str] = []
    prefer = [p.strip().lower() for p in (req.prefer_providers or []) if p and str(p).strip()] or ["auto"]

    picked = prefer[0]
    ok, why = _provider_allowed_by_session(picked, session)
    if not ok:
        notes.append(f"avoid:{picked}:{why}")
        picked = "auto"

    # Gemini‑3 posture: prefer Vertex routes when user requires gemini-3 and isn't forcing a provider.
    if req.require_model_prefix and req.require_model_prefix.startswith("gemini-3") and picked in {"auto", "gemini", "gemini-cli"}:
        notes.append("policy:gemini-3 prefers vertex routes when available (control plane decides)")
        picked = "auto"

    # Load MANIFEST for app-of-apps routing
    root = find_pluribus_root(Path.cwd())
    manifest = _load_manifest(root)
    
    # Extract domains from NL prompt
    domains = _extract_domains(req.goal, manifest)
    
    # Route to target app
    target_app = _route_to_app(domains, manifest)
    notes.append(f"routing:domains={domains}→{target_app}")

    # Persona selection (deterministic) — vocabulary/grammar for intra+extra agent routing.
    persona_id = "subagent.narrow_coder" if depth == "narrow" else "ring0.architect"
    persona_reason = "default"
    try:
        personas_path = root / "nucleus" / "specs" / "personas.json"
        if personas_path.exists():
            registry = load_personas(personas_path)
            pick = choose_persona(registry, goal=req.goal, depth=depth, kind=kind, effects=effects)
            persona_id, persona_reason = pick.persona_id, pick.reason
        else:
            notes.append("persona:missing_registry")
    except Exception:
        notes.append("persona:choose_failed")

    # LENS/LASER synthesis mode detection
    synthesis_mode = req.synthesis_mode
    lens_laser_config = None

    # Auto-detect synthesis mode for complex multi-model queries
    if synthesis_mode is None and depth == "deep" and kind in {"distill", "verify", "audit"}:
        # Suggest LENS/LASER for deep verification/distillation tasks
        notes.append("synthesis:lens_laser_candidate")

    if synthesis_mode == "lens_laser":
        # Configure LENS/LASER synthesis
        lens_laser_config = req.synthesis_config or {
            "models": ["claude", "gemini", "gpt"],
            "samples_per_model": 2,
            "budgets": {
                "conjecture_max": 0.05,
                "missing_max": 0.10,
                "cogload_max": 0.30,
                "info_min": 0.50
            }
        }
        notes.append(f"synthesis:lens_laser:models={lens_laser_config.get('models', [])}")

    return RoutePlan(
        req_id=req.req_id,
        depth=depth,
        lane=lane,
        provider=picked,
        context_mode=context_mode,
        topology=topology,
        fanout=fanout,
        topology_reason=topology_reason,
        persona_id=persona_id,
        persona_reason=persona_reason,
        domains=domains,
        target_app=target_app,
        notes=notes,
        synthesis_mode=synthesis_mode,
        lens_laser_config=lens_laser_config,
    )


def parse_laser_request(obj: dict[str, Any]) -> LASERRequest:
    req_id = str(obj.get("req_id") or uuid.uuid4())
    goal = str(obj.get("goal") or "")
    kind = str(obj.get("kind") or "distill")
    effects = str(obj.get("effects") or "none")
    prefer_providers = list(obj.get("prefer_providers") or [])
    require_model_prefix = obj.get("require_model_prefix")
    synthesis_mode = obj.get("synthesis_mode")
    synthesis_config = obj.get("synthesis_config")

    if synthesis_mode not in (None, "lens_laser"):
        synthesis_mode = None

    return LASERRequest(
        req_id=req_id,
        goal=goal,
        kind=kind,
        effects=effects,
        prefer_providers=[str(p) for p in prefer_providers],
        require_model_prefix=require_model_prefix,
        synthesis_mode=synthesis_mode,
        synthesis_config=synthesis_config,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="collimator.py", description="LASER/Collimator routing planner (deterministic; emits bus evidence).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (default: $PLURIBUS_BUS_DIR or /pluribus/.pluribus/bus).")
    p.add_argument("--root", default=None, help="Pluribus root (default: search upward).")
    p.add_argument("--actor", default=None)
    p.add_argument("--json", default=None, help="LASERRequest JSON string (or omit to read stdin).")
    p.add_argument("--no-emit", action="store_true", help="Do not emit bus evidence.")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "laser"
    root = Path(args.root).expanduser().resolve() if args.root else find_pluribus_root(Path.cwd())
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()

    raw = args.json
    if raw is None:
        raw = sys.stdin.read()
    try:
        obj = json.loads(raw) if raw and raw.strip() else {}
    except Exception:
        obj = {}

    if not isinstance(obj, dict):
        obj = {}
    req = parse_laser_request(obj)
    session = load_vps_session(root)
    plan = route_query(req, session=session)

    out = {
        "req_id": plan.req_id,
        "depth": plan.depth,
        "lane": plan.lane,
        "provider": plan.provider,
        "context_mode": plan.context_mode,
        "topology": plan.topology,
        "fanout": plan.fanout,
        "topology_reason": plan.topology_reason,
        "persona_id": plan.persona_id,
        "persona_reason": plan.persona_reason,
        "domains": plan.domains,
        "target_app": plan.target_app,
        "notes": plan.notes,
        "synthesis_mode": plan.synthesis_mode,
        "lens_laser_config": plan.lens_laser_config,
    }

    if not args.no_emit:
        emit_bus(bus_dir, topic="laser.collimator.plan", kind="artifact", level="info", actor=actor, data=out)
        emit_bus(
            bus_dir,
            topic="laser.collimator.decision",
            kind="metric",
            level="info",
            actor=actor,
            data={k: out[k] for k in ["req_id", "depth", "lane", "provider", "context_mode", "topology", "fanout", "persona_id"]},
        )

    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
