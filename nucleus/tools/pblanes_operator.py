#!/usr/bin/env python3
"""
PBLANES - Pluribus Lanes Operator
=================================

Semantic operator for multi-agent coordination visibility.
Invokable from any agent context to view lanes status.

Protocol: DKIN v23 (Lanes Integration)
Spec: nucleus/specs/lanes_protocol_v1.md

Usage:
    python3 nucleus/tools/pblanes_operator.py
    python3 nucleus/tools/pblanes_operator.py --emit-bus
    python3 nucleus/tools/pblanes_operator.py --json
    python3 nucleus/tools/pblanes_operator.py --compact
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

PLURIBUS_ROOT = os.environ.get("PLURIBUS_ROOT", "/pluribus")
STATE_PATH = os.path.join(PLURIBUS_ROOT, "nucleus", "state", "lanes.json")
BUS_PATH = os.path.join(PLURIBUS_ROOT, ".pluribus", "bus", "events.ndjson")


def load_lanes_state() -> Dict[str, Any]:
    """Load lanes state from persistent storage."""
    if not os.path.exists(STATE_PATH):
        return {
            "version": "1.0",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lanes": [],
            "agents": []
        }
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def render_wip_meter(pct: int, width: int = 20) -> str:
    """Render a WIP meter bar."""
    filled = pct * width // 100
    return "█" * filled + "░" * (width - filled)


def status_emoji(status: str) -> str:
    """Get emoji for status."""
    return {"green": "🟢", "yellow": "🟡", "red": "🔴", "blocked": "🔴"}.get(status, "⚪")


def get_parallelism_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parallelism information from lanes state."""
    active_agents = [a for a in state.get("agents", []) if a.get("status") == "active"]
    lanes_by_owner: Dict[str, List[str]] = {}

    for lane in state.get("lanes", []):
        owner = lane.get("owner", "unknown")
        lanes_by_owner.setdefault(owner, []).append(lane["id"])

    return {
        "total_agents": len(state.get("agents", [])),
        "active_agents": len(active_agents),
        "active_agent_ids": [a["id"] for a in active_agents],
        "lanes_per_agent": lanes_by_owner,
        "parallel_lanes": len(state.get("lanes", [])),
        "parallelism_factor": len(active_agents) if active_agents else 1
    }


def render_pblanes_report(state: Dict[str, Any], compact: bool = False) -> str:
    """Render the PBLANES report."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lanes = state.get("lanes", [])
    agents = state.get("agents", [])
    parallelism = get_parallelism_info(state)

    # Calculate overall progress
    if lanes:
        overall_wip = sum(l.get("wip_pct", 0) for l in lanes) // len(lanes)
    else:
        overall_wip = 0

    lines = [
        "╔══════════════════════════════════════════════════════════════════╗",
        "║                         P B L A N E S                            ║",
        "║              Multi-Agent Coordination Dashboard                  ║",
        "╚══════════════════════════════════════════════════════════════════╝",
        "",
        f"  Generated: {now}",
        f"  State: nucleus/state/lanes.json",
        f"  Protocol: DKIN v23 (Lanes Integration)",
        "",
        "┌─────────────────────────────────────────────────────────────────┐",
        "│ PARALLELISM                                                     │",
        "├─────────────────────────────────────────────────────────────────┤",
        f"│  Active Agents: {parallelism['active_agents']}/{parallelism['total_agents']}  │  Parallel Lanes: {parallelism['parallel_lanes']}  │  Factor: {parallelism['parallelism_factor']}x       │",
        "└─────────────────────────────────────────────────────────────────┘",
        "",
    ]

    # Agent membership section
    lines.extend([
        "┌─────────────────────────────────────────────────────────────────┐",
        "│ AGENT MEMBERSHIP                                                │",
        "├──────────┬──────────┬─────────────────────────────────────────┤",
        "│ Agent    │ Status   │ Assigned Lane                           │",
        "├──────────┼──────────┼─────────────────────────────────────────┤",
    ])

    for agent in agents:
        emoji = "🟢" if agent.get("status") == "active" else "⚪"
        agent_id = agent.get("id", "unknown")[:8]
        status = agent.get("status", "unknown")[:8]
        lane = agent.get("lane") or "(unassigned)"
        lane = lane[:39]
        lines.append(f"│ {agent_id:<8} │ {emoji} {status:<6} │ {lane:<39} │")

    lines.extend([
        "└──────────┴──────────┴─────────────────────────────────────────┘",
        "",
    ])

    # Lanes progress section
    lines.extend([
        "┌─────────────────────────────────────────────────────────────────┐",
        "│ LANES PROGRESS                                                  │",
        "├──────────────────────────────┬───────┬────────┬─────────────────┤",
        "│ Lane                         │ Owner │ WIP    │ Meter           │",
        "├──────────────────────────────┼───────┼────────┼─────────────────┤",
    ])

    for lane in lanes:
        emoji = status_emoji(lane.get("status", "green"))
        name = lane.get("name", lane.get("id", "unknown"))[:26]
        owner = lane.get("owner", "?")[:5]
        wip = lane.get("wip_pct", 0)
        meter = render_wip_meter(wip, 15)
        lines.append(f"│ {emoji} {name:<25} │ {owner:<5} │ {wip:>3}%   │ {meter} │")

    lines.extend([
        "├──────────────────────────────┴───────┴────────┴─────────────────┤",
        f"│ OVERALL PROGRESS: {overall_wip}%  {render_wip_meter(overall_wip, 40)}   │",
        "└─────────────────────────────────────────────────────────────────┘",
        "",
    ])

    if not compact:
        # Lane details section
        lines.extend([
            "┌─────────────────────────────────────────────────────────────────┐",
            "│ LANE DETAILS                                                    │",
            "├─────────────────────────────────────────────────────────────────┤",
        ])

        for lane in lanes:
            emoji = status_emoji(lane.get("status", "green"))
            name = lane.get("name", lane.get("id", "unknown"))
            desc = lane.get("description", "")[:55]
            commits = lane.get("commits", [])[-3:]  # Last 3 commits
            blockers = lane.get("blockers", [])
            next_actions = lane.get("next_actions", [])[:2]

            lines.append(f"│ {emoji} {name}")
            if desc:
                lines.append(f"│   Description: {desc}")
            if commits:
                lines.append(f"│   Commits: {', '.join(commits[:3])}")
            if blockers:
                lines.append(f"│   ⚠️ Blockers: {', '.join(blockers[:2])}")
            if next_actions:
                lines.append(f"│   → Next: {next_actions[0]}")
            lines.append("│")

        lines.append("└─────────────────────────────────────────────────────────────────┘")

    # Footer
    lines.extend([
        "",
        "─────────────────────────────────────────────────────────────────",
        "  PBLANES v1 │ Protocol: lanes_protocol_v1 │ DKIN v23",
        "  Tool: nucleus/tools/pblanes_operator.py",
        "─────────────────────────────────────────────────────────────────",
    ])

    return "\n".join(lines)


def emit_pblanes_event(state: Dict[str, Any], actor: str) -> Dict[str, Any]:
    """Emit PBLANES event to bus."""
    parallelism = get_parallelism_info(state)
    lanes = state.get("lanes", [])

    if lanes:
        overall_wip = sum(l.get("wip_pct", 0) for l in lanes) // len(lanes)
    else:
        overall_wip = 0

    event = {
        "id": str(uuid.uuid4()),
        "ts": int(time.time() * 1000),
        "topic": "operator.pblanes.report",
        "kind": "metric",
        "level": "info",
        "actor": actor,
        "data": {
            "overall_wip": overall_wip,
            "lane_count": len(lanes),
            "parallelism": parallelism,
            "lanes_summary": {
                l["id"]: {
                    "name": l.get("name"),
                    "status": l.get("status"),
                    "wip_pct": l.get("wip_pct"),
                    "owner": l.get("owner")
                }
                for l in lanes
            },
            "agents": [
                {"id": a["id"], "status": a.get("status"), "lane": a.get("lane")}
                for a in state.get("agents", [])
            ]
        }
    }

    os.makedirs(os.path.dirname(BUS_PATH), exist_ok=True)
    with open(BUS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    return event


def render_json_report(state: Dict[str, Any]) -> str:
    """Render lanes state as JSON."""
    parallelism = get_parallelism_info(state)
    lanes = state.get("lanes", [])

    if lanes:
        overall_wip = sum(l.get("wip_pct", 0) for l in lanes) // len(lanes)
    else:
        overall_wip = 0

    report = {
        "operator": "PBLANES",
        "version": "1.0",
        "protocol": "DKIN v23",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_wip": overall_wip,
        "parallelism": parallelism,
        "lanes": state.get("lanes", []),
        "agents": state.get("agents", [])
    }

    return json.dumps(report, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PBLANES - Multi-Agent Coordination Dashboard",
        epilog="Protocol: DKIN v23 (Lanes Integration)"
    )
    parser.add_argument("--emit-bus", action="store_true", help="Emit report to bus")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--compact", action="store_true", help="Compact output (no lane details)")
    parser.add_argument("--actor", default=os.environ.get("PLURIBUS_ACTOR", "pblanes"), help="Actor ID")
    args = parser.parse_args()

    state = load_lanes_state()

    if args.json:
        print(render_json_report(state))
    else:
        print(render_pblanes_report(state, compact=args.compact))

    if args.emit_bus:
        event = emit_pblanes_event(state, args.actor)
        print(f"\n✓ Emitted: {event['topic']} (id: {event['id'][:8]})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
