#!/usr/bin/env python3
"""
DKIN/PAIP Flow Monitor — TUI Widget for Protocol Compliance & Code Flow

A terminal-based monitor synthesizing all DKIN protocol versions (v1-v19):
- Bus event flow and health visualization
- Agent task lifecycle tracking
- Protocol compliance with remediation guidance
- Evolutionary cycle (v19) phase tracking

Usage:
    python3 dkin_flow_monitor.py [--watch] [--compact] [--emit-bus]

Bus Topic: dkin.flow_monitor.snapshot (kind=metric)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True

# ============================================================================
# Visual Elements
# ============================================================================

SPARK = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_HALF = "▒"
BAR_EMPTY = "░"
CHECK = "✓"
CROSS = "✗"
CIRCLE = "○"
LOCK = "🔒"
UNLOCK = "🔓"
ARROW_RIGHT = "→"

# ANSI Colors
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
    WHITE = "\033[37m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

# ============================================================================
# Protocol Versions
# ============================================================================

PROTOCOL_VERSIONS = [
    ("v15", "Context", ["log hygiene", "context guards"]),
    ("v16", "PBLOCK", ["milestone freeze", "feature lock"]),
    ("v17", "Hygiene", ["bus rotation", "context window safety"]),
    ("v18", "Resilience", ["Agentic State Graph", "lossless handoff"]),
    ("v19", "Evolution", ["Percolation Loop", "CMP metric", "HGT"]),
]

EVO_PHASES = ["percolate", "assimilate", "mutate", "test", "promote"]

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BusHealth:
    size_mb: float = 0.0
    event_count: int = 0
    oldest_age_hours: float = 0.0
    velocity: float = 0.0
    needs_rotation: bool = False

@dataclass
class PBLOCKState:
    active: bool = False
    milestone: str = ""
    entered_by: str = ""
    exit_criteria: dict = field(default_factory=lambda: {
        "all_tests_pass": False,
        "pushed_to_remotes": False,
        "pushed_to_github": False,
    })
    violations: int = 0

@dataclass
class PAIPClone:
    clone_dir: str = ""
    agent_id: str = ""
    branch: str = ""
    is_orphan: bool = False
    is_stale: bool = False
    uncommitted: int = 0

@dataclass
class CMPMetrics:
    utility: float = 0.0
    robustness: float = 0.0
    complexity: float = 1.0
    cost: float = 1.0

    @property
    def score(self) -> float:
        if self.complexity == 0 or self.cost == 0:
            return 0.0
        return (self.utility * self.robustness) / (self.complexity * self.cost)

@dataclass
class Compliance:
    version: str = "v19"
    compliant: bool = True
    violations: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)

@dataclass
class FlowState:
    protocol_version: str = "v19"
    evo_phase: str = "idle"
    cmp: CMPMetrics = field(default_factory=CMPMetrics)
    pblock: PBLOCKState = field(default_factory=PBLOCKState)
    bus_health: BusHealth = field(default_factory=BusHealth)
    paip_clones: list = field(default_factory=list)
    compliance: Compliance = field(default_factory=Compliance)
    recent_events: list = field(default_factory=list)
    agent_tasks: list = field(default_factory=list)

# ============================================================================
# Bus Analysis
# ============================================================================

def load_bus_events(bus_dir: Path, window_s: int = 900) -> list[dict]:
    """Load recent bus events."""
    events_path = bus_dir / "events.ndjson"
    if not events_path.exists():
        return []

    events = []
    now = time.time()
    cutoff = now - window_s

    try:
        with open(events_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    ts = e.get("ts", 0)
                    if ts >= cutoff:
                        events.append(e)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return sorted(events, key=lambda x: x.get("ts", 0), reverse=True)

def analyze_flow_state(events: list[dict], bus_dir: Path) -> FlowState:
    """Analyze events to build flow state."""
    state = FlowState()

    # Bus health
    events_path = bus_dir / "events.ndjson"
    if events_path.exists():
        try:
            size_bytes = events_path.stat().st_size
            state.bus_health.size_mb = size_bytes / (1024 * 1024)
            state.bus_health.event_count = len(events)
            if events:
                oldest = min(e.get("ts", time.time()) for e in events)
                state.bus_health.oldest_age_hours = (time.time() - oldest) / 3600
                state.bus_health.velocity = len(events) / (state.bus_health.oldest_age_hours or 1)
            state.bus_health.needs_rotation = state.bus_health.size_mb > 100
        except Exception:
            pass

    # PBLOCK state
    pblock_events = [e for e in events if e.get("topic", "").startswith("operator.pblock")]
    if pblock_events:
        latest = pblock_events[0]
        data = latest.get("data", {})
        state.pblock.active = data.get("active", False)
        state.pblock.milestone = data.get("milestone", "")
        state.pblock.entered_by = data.get("entered_by", "")
        state.pblock.exit_criteria = {
            "all_tests_pass": data.get("all_tests_pass", False),
            "pushed_to_remotes": data.get("pushed_to_remotes", False),
            "pushed_to_github": data.get("pushed_to_github", False),
        }
        state.pblock.violations = data.get("violations", 0)

    # PAIP clones
    paip_events = [e for e in events if e.get("topic", "").startswith("paip.")]
    clone_map = {}
    for e in paip_events:
        data = e.get("data", {})
        clone_dir = data.get("clone_dir")
        if clone_dir:
            if e.get("topic") == "paip.clone.deleted":
                clone_map.pop(clone_dir, None)
            else:
                clone_map[clone_dir] = PAIPClone(
                    clone_dir=clone_dir,
                    agent_id=data.get("agent_id", "unknown"),
                    branch=data.get("branch", ""),
                    is_orphan=e.get("topic") == "paip.orphan.detected",
                    is_stale=data.get("stale", False),
                    uncommitted=data.get("uncommitted", 0),
                )
    state.paip_clones = list(clone_map.values())

    # Evolutionary phase detection
    evo_events = [e for e in events if e.get("topic", "").startswith("evolution.")]
    if evo_events:
        latest = evo_events[0]
        state.evo_phase = latest.get("data", {}).get("phase", "idle")

    # CMP metrics
    cmp_events = [e for e in events if e.get("topic", "").startswith("cmp.")]
    if cmp_events:
        latest = cmp_events[0]
        data = latest.get("data", {})
        state.cmp = CMPMetrics(
            utility=data.get("utility", 0),
            robustness=data.get("robustness", 0),
            complexity=data.get("complexity", 1),
            cost=data.get("cost", 1),
        )

    # Task lifecycle (v18)
    task_events = [e for e in events if "task" in e.get("topic", "")]
    task_map = {}
    for e in task_events:
        data = e.get("data", {})
        task_id = data.get("task_id")
        if task_id:
            task_map[task_id] = {
                "task_id": task_id,
                "species": data.get("species", "unknown"),
                "state": data.get("status", "PENDING"),
                "progress": data.get("progress", 0),
            }
    state.agent_tasks = list(task_map.values())

    # Compliance check
    violations = []
    recommendations = []

    if state.bus_health.needs_rotation:
        violations.append("Bus exceeds 100MB threshold")
        recommendations.append("Run: pbhygiene --rotate-bus")

    if any(c.is_orphan for c in state.paip_clones):
        violations.append("Orphan PAIP clones detected")
        recommendations.append("Clean orphans before PBLOCK")

    if state.pblock.active and state.pblock.violations > 0:
        violations.append(f"{state.pblock.violations} PBLOCK violations")
        recommendations.append("Use fix:/test:/refactor: prefixes only")

    state.compliance = Compliance(
        version="v19",
        compliant=len(violations) == 0,
        violations=violations,
        recommendations=recommendations,
    )

    state.recent_events = events[:20]

    return state

# ============================================================================
# Rendering
# ============================================================================

def sparkline(values: list[int], width: int = 20) -> str:
    """Generate ASCII sparkline."""
    if not values:
        return " " * width
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return SPARK[0] * len(values)
    out = []
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(SPARK) - 1))
        out.append(SPARK[max(0, min(len(SPARK) - 1, idx))])
    return "".join(out)

def progress_bar(value: float, width: int = 20) -> str:
    """Generate progress bar."""
    filled = int(value * width)
    return BAR_FULL * filled + BAR_EMPTY * (width - filled)

def render_header(state: FlowState) -> str:
    """Render header with protocol timeline."""
    lines = []
    lines.append(f"{C.BOLD}╔══════════════════════════════════════════════════════════════════════╗{C.RESET}")
    lines.append(f"{C.BOLD}║  DKIN/PAIP Flow Monitor — Protocol v{state.protocol_version}                          ║{C.RESET}")
    lines.append(f"{C.BOLD}╠══════════════════════════════════════════════════════════════════════╣{C.RESET}")

    # Protocol timeline
    timeline = []
    for ver, name, _ in PROTOCOL_VERSIONS:
        if ver == state.protocol_version:
            timeline.append(f"{C.CYAN}{C.BOLD}[{ver}]{C.RESET}")
        else:
            timeline.append(f"{C.DIM}{ver}{C.RESET}")
    lines.append(f"║  Protocol: {' → '.join(timeline)}{' ' * 10}║")

    return "\n".join(lines)

def render_compliance(state: FlowState) -> str:
    """Render compliance panel."""
    comp = state.compliance
    lines = []

    if comp.compliant:
        lines.append(f"║  {C.GREEN}{CHECK} COMPLIANT{C.RESET} — Protocol {comp.version}{' ' * 40}║")
    else:
        lines.append(f"║  {C.RED}{CROSS} NON-COMPLIANT{C.RESET} — Protocol {comp.version}{' ' * 36}║")
        for v in comp.violations[:2]:
            lines.append(f"║    {C.RED}• {v[:55]}{C.RESET}{' ' * max(0, 60 - len(v))}║")
        for r in comp.recommendations[:2]:
            lines.append(f"║    {C.YELLOW}→ {r[:55]}{C.RESET}{' ' * max(0, 60 - len(r))}║")

    return "\n".join(lines)

def render_pblock(state: FlowState) -> str:
    """Render PBLOCK status."""
    pb = state.pblock
    lines = []

    if pb.active:
        lines.append(f"╠──────────────────────────────────────────────────────────────────────╣")
        lines.append(f"║  {C.YELLOW}{LOCK} PBLOCK ACTIVE{C.RESET} — {pb.milestone[:40]}{' ' * max(0, 30 - len(pb.milestone))}║")
        ec = pb.exit_criteria
        tests = f"{C.GREEN}{CHECK}{C.RESET}" if ec.get("all_tests_pass") else f"{C.DIM}{CIRCLE}{C.RESET}"
        remote = f"{C.GREEN}{CHECK}{C.RESET}" if ec.get("pushed_to_remotes") else f"{C.DIM}{CIRCLE}{C.RESET}"
        github = f"{C.GREEN}{CHECK}{C.RESET}" if ec.get("pushed_to_github") else f"{C.DIM}{CIRCLE}{C.RESET}"
        lines.append(f"║    Exit: {tests} Tests  {remote} Remotes  {github} GitHub{' ' * 30}║")
        if pb.violations > 0:
            lines.append(f"║    {C.RED}⚠ {pb.violations} guard violations{C.RESET}{' ' * 45}║")
    else:
        lines.append(f"╠──────────────────────────────────────────────────────────────────────╣")
        lines.append(f"║  {C.GREEN}{UNLOCK} Normal Development{C.RESET}{' ' * 48}║")

    return "\n".join(lines)

def render_bus_health(state: FlowState) -> str:
    """Render bus health metrics."""
    bh = state.bus_health
    lines = []

    color = C.GREEN if not bh.needs_rotation else C.YELLOW if bh.size_mb < 200 else C.RED
    lines.append(f"╠──────────────────────────────────────────────────────────────────────╣")
    lines.append(f"║  Bus Health: {color}{bh.size_mb:.1f}MB{C.RESET} | {bh.event_count} events | {bh.velocity:.0f}/hr{' ' * 25}║")

    if bh.needs_rotation:
        lines.append(f"║    {C.YELLOW}⟳ Rotation recommended{C.RESET}{' ' * 44}║")

    return "\n".join(lines)

def render_paip(state: FlowState) -> str:
    """Render PAIP clone status."""
    clones = state.paip_clones
    lines = []

    lines.append(f"╠──────────────────────────────────────────────────────────────────────╣")
    lines.append(f"║  PAIP Isolation: {len(clones)} clone(s){' ' * 45}║")

    active = [c for c in clones if not c.is_orphan and not c.is_stale]
    problems = [c for c in clones if c.is_orphan or c.is_stale]

    for c in active[:2]:
        lines.append(f"║    {C.GREEN}●{C.RESET} {c.agent_id[:15]} → {c.branch[:30]}{' ' * 20}║")

    for c in problems[:2]:
        icon = "👻" if c.is_orphan else "⏰"
        lines.append(f"║    {C.YELLOW}{icon}{C.RESET} {c.agent_id[:15]} (cleanup needed){' ' * 28}║")

    return "\n".join(lines)

def render_evo_cycle(state: FlowState) -> str:
    """Render evolutionary cycle (v19)."""
    lines = []

    lines.append(f"╠──────────────────────────────────────────────────────────────────────╣")
    lines.append(f"║  Evolutionary Cycle (v19):{' ' * 42}║")

    phases_display = []
    for p in EVO_PHASES:
        if p == state.evo_phase:
            phases_display.append(f"{C.CYAN}{C.BOLD}[{p}]{C.RESET}")
        else:
            phases_display.append(f"{C.DIM}{p}{C.RESET}")

    lines.append(f"║    {' → '.join(phases_display)}{' ' * 10}║")

    # CMP gauge
    cmp = state.cmp
    score = cmp.score
    bar = progress_bar(min(1.0, score), 20)
    color = C.GREEN if score > 0.7 else C.YELLOW if score > 0.4 else C.RED
    lines.append(f"║    CMP: {color}{bar}{C.RESET} {score*100:.1f}%{' ' * 30}║")
    lines.append(f"║    U:{cmp.utility:.2f} R:{cmp.robustness:.2f} C:{cmp.complexity:.1f} $:{cmp.cost:.2f}{' ' * 30}║")

    return "\n".join(lines)

def render_tasks(state: FlowState) -> str:
    """Render agent task flow (v18)."""
    tasks = state.agent_tasks
    lines = []

    lines.append(f"╠──────────────────────────────────────────────────────────────────────╣")
    running = [t for t in tasks if t.get("state") == "RUNNING"]
    lines.append(f"║  Task Flow (v18): {len(running)} active / {len(tasks)} total{' ' * 35}║")

    for t in tasks[:4]:
        state_color = {
            "RUNNING": C.BLUE,
            "COMPLETED": C.GREEN,
            "FAILED": C.RED,
            "PENDING": C.DIM,
        }.get(t.get("state", ""), C.DIM)
        progress = t.get("progress", 0)
        bar = progress_bar(progress, 10)
        lines.append(f"║    {state_color}●{C.RESET} {t.get('species', 'unk')[:8]:8} {bar} {progress*100:.0f}%{' ' * 30}║")

    return "\n".join(lines)

def render_events(state: FlowState) -> str:
    """Render recent DKIN events."""
    events = state.recent_events
    lines = []

    lines.append(f"╠──────────────────────────────────────────────────────────────────────╣")
    lines.append(f"║  Recent Events:{' ' * 53}║")

    for e in events[:5]:
        ts = time.strftime("%H:%M:%S", time.localtime(e.get("ts", 0)))
        topic = e.get("topic", "")[:35]
        actor = e.get("actor", "")[:10]
        lines.append(f"║    {C.DIM}{ts}{C.RESET} {topic:35} {C.CYAN}{actor}{C.RESET}{' ' * 5}║")

    return "\n".join(lines)

def render_footer() -> str:
    """Render footer."""
    return f"╚══════════════════════════════════════════════════════════════════════╝"

def render_full(state: FlowState) -> str:
    """Render full monitor output."""
    sections = [
        render_header(state),
        render_compliance(state),
        render_pblock(state),
        render_bus_health(state),
        render_paip(state),
        render_evo_cycle(state),
        render_tasks(state),
        render_events(state),
        render_footer(),
    ]
    return "\n".join(sections)

def render_compact(state: FlowState) -> str:
    """Render compact one-line summary."""
    pb_icon = LOCK if state.pblock.active else UNLOCK
    comp_icon = CHECK if state.compliance.compliant else CROSS
    evo = state.evo_phase[:3].upper()
    cmp_score = state.cmp.score * 100

    return (
        f"{comp_icon} v{state.protocol_version} | "
        f"{pb_icon} PBLOCK | "
        f"Bus:{state.bus_health.size_mb:.0f}MB | "
        f"PAIP:{len(state.paip_clones)} | "
        f"EVO:{evo} | "
        f"CMP:{cmp_score:.0f}%"
    )

# ============================================================================
# Bus Emission
# ============================================================================

def emit_snapshot(state: FlowState, bus_dir: Path) -> None:
    """Emit flow monitor snapshot to bus."""
    events_path = bus_dir / "events.ndjson"
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        import uuid
        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": "dkin.flow_monitor.snapshot",
            "kind": "metric",
            "actor": os.environ.get("PLURIBUS_ACTOR", "dkin_monitor"),
            "level": "info",
            "data": {
                "protocol_version": state.protocol_version,
                "compliant": state.compliance.compliant,
                "pblock_active": state.pblock.active,
                "bus_size_mb": state.bus_health.size_mb,
                "paip_clones": len(state.paip_clones),
                "evo_phase": state.evo_phase,
                "cmp_score": state.cmp.score,
                "violations": state.compliance.violations,
            },
        }
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"Warning: Could not emit to bus: {e}", file=sys.stderr)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DKIN/PAIP Flow Monitor")
    parser.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    parser.add_argument("--window", type=int, default=900, help="Event window in seconds")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--compact", action="store_true", help="Compact one-line output")
    parser.add_argument("--emit-bus", action="store_true", help="Emit snapshot to bus")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval in seconds")
    args = parser.parse_args()

    bus_dir = Path(args.bus_dir)

    def run_once():
        events = load_bus_events(bus_dir, args.window)
        state = analyze_flow_state(events, bus_dir)

        if args.emit_bus:
            emit_snapshot(state, bus_dir)

        if args.compact:
            print(render_compact(state))
        else:
            print(render_full(state))

    if args.watch:
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                run_once()
                print(f"\n{C.DIM}Refreshing every {args.interval}s... (Ctrl+C to exit){C.RESET}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        run_once()

if __name__ == "__main__":
    main()
