#!/usr/bin/env python3
"""
Lanes Report Tool - Multi-Agent Coordination Tracking

Protocol: nucleus/specs/lanes_protocol_v1.md
State: nucleus/state/lanes.json
Schema: nucleus/specs/lanes_report_schema.json

Usage:
    python3 lanes_report.py --status              # Show current lanes
    python3 lanes_report.py --update LANE --wip N # Update lane WIP
    python3 lanes_report.py --emit                # Emit to bus
    python3 lanes_report.py --render              # Markdown report
    python3 lanes_report.py --add-lane ID NAME OWNER  # Add new lane
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

# Resolve paths relative to PLURIBUS_ROOT
PLURIBUS_ROOT = os.environ.get("PLURIBUS_ROOT", "/pluribus")
STATE_PATH = os.path.join(PLURIBUS_ROOT, "nucleus", "state", "lanes.json")
BUS_PATH = os.path.join(PLURIBUS_ROOT, ".pluribus", "bus", "events.ndjson")


def load_state() -> Dict[str, Any]:
    """Load lanes state from JSON file."""
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


def save_state(state: Dict[str, Any]) -> None:
    """Save lanes state to JSON file."""
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def render_wip_meter(pct: int, width: int = 20) -> str:
    """Render a WIP meter bar."""
    filled = pct * width // 100
    return "█" * filled + "░" * (width - filled)


def status_emoji(status: str) -> str:
    """Get emoji for status."""
    return {"green": "🟢", "yellow": "🟡", "red": "🔴", "blocked": "🔴"}.get(status, "⚪")


def update_lane(
    lane_id: str,
    wip_pct: Optional[int] = None,
    status: Optional[str] = None,
    note: Optional[str] = None,
    commit: Optional[str] = None,
    blocker: Optional[str] = None,
    next_action: Optional[str] = None,
) -> Dict[str, Any]:
    """Update a lane's state."""
    state = load_state()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    lane = next((l for l in state["lanes"] if l["id"] == lane_id), None)
    if not lane:
        raise ValueError(f"Lane not found: {lane_id}")

    if wip_pct is not None:
        lane["wip_pct"] = max(0, min(100, wip_pct))
    if status is not None:
        lane["status"] = status
    if commit:
        lane.setdefault("commits", []).append(commit)
    if blocker:
        lane.setdefault("blockers", []).append(blocker)
    if next_action:
        lane.setdefault("next_actions", []).append(next_action)

    # Add history entry
    if wip_pct is not None or note:
        lane.setdefault("history", []).append({
            "ts": now,
            "wip_pct": lane["wip_pct"],
            "note": note or f"Updated to {lane['wip_pct']}%"
        })

    save_state(state)
    return lane


def add_lane(
    lane_id: str,
    name: str,
    owner: str,
    description: str = "",
    wip_pct: int = 0,
) -> Dict[str, Any]:
    """Add a new lane."""
    state = load_state()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if any(l["id"] == lane_id for l in state["lanes"]):
        raise ValueError(f"Lane already exists: {lane_id}")

    lane = {
        "id": lane_id,
        "name": name,
        "status": "green",
        "wip_pct": wip_pct,
        "owner": owner,
        "description": description,
        "commits": [],
        "blockers": [],
        "next_actions": [],
        "history": [{"ts": now, "wip_pct": wip_pct, "note": "Lane created"}]
    }
    state["lanes"].append(lane)
    save_state(state)
    return lane


def update_agent(agent_id: str, status: str = "active", lane: Optional[str] = None) -> None:
    """Update agent status."""
    state = load_state()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    agent = next((a for a in state["agents"] if a["id"] == agent_id), None)
    if agent:
        agent["status"] = status
        agent["lane"] = lane
        agent["last_seen"] = now
    else:
        state["agents"].append({
            "id": agent_id,
            "status": status,
            "lane": lane,
            "last_seen": now
        })
    save_state(state)


def emit_lanes_state(actor: str = "lanes-tool") -> Dict[str, Any]:
    """Emit lanes state to bus."""
    state = load_state()

    event = {
        "id": str(uuid.uuid4()),
        "ts": int(time.time() * 1000),
        "topic": "operator.lanes.state",
        "kind": "metric",
        "level": "info",
        "actor": actor,
        "data": {
            "lane_count": len(state["lanes"]),
            "agent_count": len(state["agents"]),
            "overall_wip": sum(l["wip_pct"] for l in state["lanes"]) // max(len(state["lanes"]), 1),
            "lanes_summary": {
                l["id"]: {"status": l["status"], "wip_pct": l["wip_pct"], "owner": l["owner"]}
                for l in state["lanes"]
            }
        }
    }

    os.makedirs(os.path.dirname(BUS_PATH), exist_ok=True)
    with open(BUS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    return event


def render_lanes_markdown() -> str:
    """Render lanes as markdown report."""
    state = load_state()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    lines = [
        "# LANES REPORT",
        f"**Generated**: {now}",
        f"**State**: `nucleus/state/lanes.json`",
        f"**Agents**: {len([a for a in state['agents'] if a['status'] == 'active'])} active",
        "",
        "---",
        "",
        "## Lane Status",
        "",
        "| Lane | Owner | WIP | Meter |",
        "|------|-------|-----|-------|",
    ]

    for lane in state["lanes"]:
        emoji = status_emoji(lane["status"])
        meter = render_wip_meter(lane["wip_pct"])
        lines.append(f"| {emoji} {lane['name']} | {lane['owner']} | {lane['wip_pct']}% | `{meter}` |")

    overall = sum(l["wip_pct"] for l in state["lanes"]) // max(len(state["lanes"]), 1)
    lines.extend([
        "",
        f"**Overall: {overall}%**",
        "",
        "---",
        "",
        "## Agents",
        "",
        "| Agent | Status | Lane |",
        "|-------|--------|------|",
    ])

    for agent in state["agents"]:
        emoji = "🟢" if agent["status"] == "active" else "⚪"
        lane = agent.get("lane") or "(unassigned)"
        lines.append(f"| {agent['id']} | {emoji} {agent['status']} | {lane} |")

    return "\n".join(lines)


def render_lanes_console() -> None:
    """Render lanes to console with colors."""
    state = load_state()

    print("LANES STATUS")
    print("-" * 60)

    for lane in state["lanes"]:
        emoji = status_emoji(lane["status"])
        meter = render_wip_meter(lane["wip_pct"])
        print(f"{emoji} {lane['name']:<30} {lane['wip_pct']:>3}% {meter}")

    print("-" * 60)
    overall = sum(l["wip_pct"] for l in state["lanes"]) // max(len(state["lanes"]), 1)
    print(f"Overall: {overall}%")
    print()

    print("AGENTS")
    print("-" * 40)
    for agent in state["agents"]:
        emoji = "🟢" if agent["status"] == "active" else "⚪"
        lane = agent.get("lane") or "(unassigned)"
        print(f"{emoji} {agent['id']:<12} → {lane}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lanes Report Tool")
    parser.add_argument("--status", action="store_true", help="Show current lanes")
    parser.add_argument("--update", metavar="LANE", help="Update lane by ID")
    parser.add_argument("--wip", type=int, help="Set WIP percentage (0-100)")
    parser.add_argument("--note", help="Add history note")
    parser.add_argument("--commit", help="Add commit hash")
    parser.add_argument("--set-status", choices=["green", "yellow", "red", "blocked"])
    parser.add_argument("--emit", action="store_true", help="Emit state to bus")
    parser.add_argument("--render", action="store_true", help="Render markdown")
    parser.add_argument("--add-lane", nargs=3, metavar=("ID", "NAME", "OWNER"))
    parser.add_argument("--actor", default="lanes-tool", help="Actor ID for bus events")
    args = parser.parse_args()

    if args.add_lane:
        lane = add_lane(args.add_lane[0], args.add_lane[1], args.add_lane[2])
        print(f"Added lane: {lane['id']}")

    if args.update:
        lane = update_lane(
            args.update,
            wip_pct=args.wip,
            status=args.set_status,
            note=args.note,
            commit=args.commit,
        )
        print(f"Updated {lane['id']}: {lane['wip_pct']}%")

    if args.emit:
        event = emit_lanes_state(actor=args.actor)
        print(f"Emitted: {event['topic']} (overall: {event['data']['overall_wip']}%)")

    if args.render:
        print(render_lanes_markdown())
    elif args.status or not any([args.update, args.emit, args.add_lane]):
        render_lanes_console()

    return 0


if __name__ == "__main__":
    sys.exit(main())
