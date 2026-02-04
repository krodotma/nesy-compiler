#!/usr/bin/env python3
"""
pbrecruit_operator.py - PBRECRUIT SemOp for Agent Recruitment

Automatically recruits N agents from the available pool using PBNOTIFY.
Emits swarm.recruit.request events and tracks recruitment status.

Usage:
    python3 pbrecruit_operator.py --count 5 --task "Phase 0 Evolution"
    python3 pbrecruit_operator.py --count 3 --class core --priority P0

Ring: 1 (Operator)
Protocol: DKIN v28 | PAIP v15 | Citizen v1
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


# Default agent pool (from agent_notify_class_map.json)
DEFAULT_AGENT_POOL = ["claude", "codex", "gemini", "qwen", "grok"]
EXTENDED_POOL = ["claude", "codex", "codex-beta", "gemini", "qwen", "grok", "sagent"]


def emit_bus_event(bus_path: Path, topic: str, data: Dict[str, Any], 
                   actor: str = "pbrecruit", level: str = "info") -> str:
    """Emit event directly to bus."""
    event_id = uuid.uuid4().hex
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "kind": "request",
        "level": level,
        "actor": actor,
        "data": data
    }
    bus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bus_path, 'a') as f:
        f.write(json.dumps(event) + '\n')
    return event_id


def call_pbnotify(message: str, targets: List[str], topic: str, 
                  data: Dict[str, Any], actor: str = "pbrecruit") -> Optional[str]:
    """Call pbnotify_operator.py to send notifications."""
    cmd = [
        sys.executable, "nucleus/tools/pbnotify_operator.py",
        "--message", message,
        "--data", json.dumps(data),
        "--topic", topic,
        "--actor", actor
    ]
    
    # Add each target
    for target in targets:
        cmd.extend(["--target", target])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()  # Returns the event UUID
        else:
            print(f"⚠️ pbnotify failed: {result.stderr}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"❌ Error calling pbnotify: {e}", file=sys.stderr)
        return None


def recruit_agents(count: int, 
                   task: str,
                   priority: str = "P1",
                   agent_class: str = "core",
                   deadline: str = None,
                   bus_dir: str = None) -> Dict[str, Any]:
    """
    Recruit N agents for a task.
    
    Args:
        count: Number of agents to recruit
        task: Task description
        priority: Priority level (P0, P1, P2)
        agent_class: Agent class from class map
        deadline: Optional deadline
        bus_dir: Bus directory path
    
    Returns:
        Recruitment result with assigned agents
    """
    bus_path = Path(bus_dir or ".pluribus/bus") / "events.ndjson"
    
    # Select agents from pool
    pool = EXTENDED_POOL if agent_class == "all" else DEFAULT_AGENT_POOL
    selected = pool[:min(count, len(pool))]
    
    recruitment_id = f"RECRUIT-{uuid.uuid4().hex[:8].upper()}"
    
    # Prepare recruitment data
    recruitment_data = {
        "recruitment_id": recruitment_id,
        "task": task,
        "priority": priority,
        "deadline": deadline or "ASAP",
        "agent_count": len(selected),
        "agents": selected,
        "status": "dispatched",
        "requested_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Emit recruitment request to bus
    event_id = emit_bus_event(
        bus_path,
        "swarm.recruit.request",
        recruitment_data,
        actor="pbrecruit"
    )
    
    # Send notifications via pbnotify
    notify_message = f"RECRUITMENT {recruitment_id}: {task}"
    notify_data = {
        "recruitment_id": recruitment_id,
        "task": task,
        "priority": priority,
        "deadline": deadline or "ASAP",
        "action": "report_for_duty"
    }
    
    notify_result = call_pbnotify(
        message=notify_message,
        targets=selected,
        topic="swarm.recruit.notify",
        data=notify_data
    )
    
    # Update result
    recruitment_data["bus_event_id"] = event_id
    recruitment_data["notify_event_id"] = notify_result
    recruitment_data["status"] = "notified" if notify_result else "bus_only"
    
    return recruitment_data


def main():
    parser = argparse.ArgumentParser(
        description="PBRECRUIT - Recruit agents for tasks via PBNOTIFY"
    )
    parser.add_argument(
        "--count", "-n", type=int, default=5,
        help="Number of agents to recruit (default: 5)"
    )
    parser.add_argument(
        "--task", "-t", required=True,
        help="Task description for recruited agents"
    )
    parser.add_argument(
        "--priority", "-p", default="P1",
        choices=["P0", "P1", "P2", "P3"],
        help="Priority level (default: P1)"
    )
    parser.add_argument(
        "--class", dest="agent_class", default="core",
        choices=["core", "all", "ring0", "subagents"],
        help="Agent class to recruit from (default: core)"
    )
    parser.add_argument(
        "--deadline", "-d",
        help="Optional deadline (e.g., '2hrs', 'EOD')"
    )
    parser.add_argument(
        "--bus-dir",
        help="Bus directory path"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output result as JSON"
    )
    
    args = parser.parse_args()
    
    result = recruit_agents(
        count=args.count,
        task=args.task,
        priority=args.priority,
        agent_class=args.agent_class,
        deadline=args.deadline,
        bus_dir=args.bus_dir
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"✅ Recruited {result['agent_count']} agents")
        print(f"   ID: {result['recruitment_id']}")
        print(f"   Task: {result['task']}")
        print(f"   Agents: {', '.join(result['agents'])}")
        print(f"   Priority: {result['priority']}")
        print(f"   Status: {result['status']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
