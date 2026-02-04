#!/usr/bin/env python3
"""
Crush Agent - Pluribus-Native LLM Interface

A fully integrated Pluribus agent using Charmbracelet mods/crush CLI.
Unlike the basic adapter, this agent:
- Has full knowledge of Pluribus architecture
- Can query and subscribe to bus events
- Implements A2A protocol (negotiate, decline, redirect)
- Is DKIN/PAIP protocol aware
- Can invoke semops operators
- Participates in multi-agent coordination

DKIN Protocol: v19
PAIP: v12.1 compliant
A2A: Full negotiation support
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, List, Dict
from dataclasses import dataclass, asdict

# =============================================================================
# Configuration
# =============================================================================

BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/var/lib/pluribus/.pluribus/bus"))
EVENTS_FILE = BUS_DIR / "events.ndjson"
CRUSH_BIN = os.environ.get("CRUSH_BIN", "/usr/local/bin/crush")
SEMOPS_PATH = Path("/pluribus/nucleus/specs/semops.json")
PROTOCOL_VERSION = "v19"
PAIP_VERSION = "v12.1"
A2A_VERSION = "v1"

# Agent identity
AGENT_ID = os.environ.get("PLURIBUS_AGENT_ID", f"crush-{uuid.uuid4().hex[:8]}")
AGENT_SPECIES = "crush"
ACTOR = os.environ.get("PLURIBUS_ACTOR", f"crush/{AGENT_ID}")

# =============================================================================
# Pluribus System Context (injected into all prompts)
# =============================================================================

PLURIBUS_SYSTEM_CONTEXT = """
You are operating within the Pluribus multi-agent system. Key facts:

## Architecture
- Pluribus is a neurosymbolic AI orchestration platform
- Communication happens via an append-only event bus (NDJSON)
- Agents coordinate via DKIN (Dashboard Kernel In Nucleus) protocol v19
- File isolation uses PAIP (Parallel Agent Isolation Protocol) v12.1

## Bus Topics You Should Know
- ckin.report: Agent check-in status
- operator.*.request/response: Semops operator calls
- a2a.negotiate.*: Agent-to-agent negotiation
- paip.clone.*: Filesystem isolation events
- crush.*: Your own session events

## Semops Operators Available
- CKIN: Status dashboard
- ITERATE/OITERATE: Coordination loops
- PBFLUSH: Flush context
- PBLOCK: Milestone freeze
- PBHYGIENE: System cleanup
- PBREALITY: Reality checks
- CRUSH: LLM queries (you)

## When Asked About Pluribus
You can explain: bus architecture, DKIN protocol, PAIP isolation, semops operators,
multi-agent coordination, lens/collimator routing, and agent lifecycle.

## A2A Protocol
If another agent requests your help:
- agree: Accept and process
- negotiate: Counter-propose terms
- decline: Reject with reason
- redirect: Suggest another agent

## Current Context
- Agent ID: {agent_id}
- Protocol: DKIN {protocol_version}
- PAIP: {paip_version}
- Bus: {bus_path}
"""

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BusEvent:
    ts: int
    iso: str
    topic: str
    actor: str
    level: str
    data: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class A2ARequest:
    req_id: str
    initiator: str
    target: str
    intent: str
    constraints: Dict[str, Any]
    context: Dict[str, Any]

@dataclass
class A2AResponse:
    req_id: str
    decision: str  # agree, negotiate, decline, redirect
    reason: Optional[str] = None
    counter_proposal: Optional[Dict[str, Any]] = None
    redirect_to: Optional[str] = None

# =============================================================================
# Bus Operations
# =============================================================================

def emit_event(topic: str, data: dict, level: str = "info", req_id: Optional[str] = None) -> BusEvent:
    """Emit a bus event."""
    now = datetime.now(timezone.utc)
    event = BusEvent(
        ts=int(now.timestamp() * 1000),
        iso=now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        topic=topic,
        actor=ACTOR,
        level=level,
        data={
            **data,
            "protocol_version": PROTOCOL_VERSION,
            "agent_id": AGENT_ID,
        }
    )
    if req_id:
        event.data["req_id"] = req_id

    try:
        EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_FILE, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
    except Exception as e:
        print(f"[crush_agent] bus emit failed: {e}", file=sys.stderr)

    return event


def query_bus(
    topic_prefix: Optional[str] = None,
    actor_prefix: Optional[str] = None,
    since_ms: Optional[int] = None,
    limit: int = 100
) -> List[BusEvent]:
    """Query recent bus events."""
    events = []
    try:
        if not EVENTS_FILE.exists():
            return []

        # Read last N lines efficiently
        with open(EVENTS_FILE, "rb") as f:
            # Seek to end and read backwards
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 1024 * 1024)  # Max 1MB
            f.seek(max(0, size - read_size))
            lines = f.read().decode("utf-8", errors="ignore").strip().split("\n")

        for line in reversed(lines[-500:]):  # Last 500 events max
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                if topic_prefix and not e.get("topic", "").startswith(topic_prefix):
                    continue
                if actor_prefix and not e.get("actor", "").startswith(actor_prefix):
                    continue
                if since_ms and e.get("ts", 0) < since_ms:
                    continue
                events.append(BusEvent(**{k: e.get(k) for k in ["ts", "iso", "topic", "actor", "level", "data"]}))
                if len(events) >= limit:
                    break
            except (json.JSONDecodeError, TypeError):
                continue
    except Exception as e:
        print(f"[crush_agent] bus query failed: {e}", file=sys.stderr)

    return list(reversed(events))  # Chronological order


def get_bus_summary() -> dict:
    """Get a summary of recent bus activity."""
    events = query_bus(limit=200)
    if not events:
        return {"event_count": 0, "topics": [], "actors": []}

    topics = {}
    actors = {}
    for e in events:
        topics[e.topic.split(".")[0]] = topics.get(e.topic.split(".")[0], 0) + 1
        actor_base = e.actor.split("/")[0] if e.actor else "unknown"
        actors[actor_base] = actors.get(actor_base, 0) + 1

    return {
        "event_count": len(events),
        "time_range_ms": events[-1].ts - events[0].ts if len(events) > 1 else 0,
        "topics": sorted(topics.items(), key=lambda x: -x[1])[:10],
        "actors": sorted(actors.items(), key=lambda x: -x[1])[:10],
    }

# =============================================================================
# A2A Protocol Implementation
# =============================================================================

def handle_a2a_request(request: A2ARequest) -> A2AResponse:
    """Handle an incoming A2A negotiation request."""
    emit_event("a2a.negotiate.received", {
        "req_id": request.req_id,
        "initiator": request.initiator,
        "intent": request.intent,
    }, req_id=request.req_id)

    # Check capabilities
    my_capabilities = ["query", "review", "explain", "refactor", "chat"]
    required = request.constraints.get("required_capabilities", [])

    missing = [c for c in required if c not in my_capabilities]
    if missing:
        response = A2AResponse(
            req_id=request.req_id,
            decision="decline",
            reason=f"Missing capabilities: {missing}"
        )
    elif request.intent in ["query", "review", "explain", "refactor"]:
        response = A2AResponse(
            req_id=request.req_id,
            decision="agree"
        )
    else:
        # Negotiate or redirect
        response = A2AResponse(
            req_id=request.req_id,
            decision="negotiate",
            counter_proposal={"supported_intents": my_capabilities}
        )

    emit_event("a2a.negotiate.response", {
        "req_id": request.req_id,
        "decision": response.decision,
        "reason": response.reason,
    }, req_id=request.req_id)

    return response


def send_a2a_request(
    target: str,
    intent: str,
    context: dict,
    constraints: Optional[dict] = None
) -> str:
    """Send an A2A request to another agent."""
    req_id = str(uuid.uuid4())[:8]

    emit_event("a2a.negotiate.request", {
        "req_id": req_id,
        "initiator": ACTOR,
        "target": target,
        "intent": intent,
        "constraints": constraints or {},
        "context": context,
    }, req_id=req_id)

    return req_id

# =============================================================================
# DKIN/PAIP Protocol Awareness
# =============================================================================

def get_dkin_state() -> dict:
    """Get current DKIN protocol state from bus."""
    state = {
        "protocol_version": PROTOCOL_VERSION,
        "paip_version": PAIP_VERSION,
        "pblock_active": False,
        "active_agents": [],
        "recent_operators": [],
    }

    events = query_bus(limit=100)
    seen_agents = set()
    seen_ops = []

    for e in events:
        if e.actor:
            seen_agents.add(e.actor.split("/")[0])

        if e.topic == "operator.pblock.enter":
            state["pblock_active"] = True
            state["pblock_milestone"] = e.data.get("milestone")
        elif e.topic == "operator.pblock.exit":
            state["pblock_active"] = False

        if e.topic.startswith("operator.") and ".request" in e.topic:
            op = e.topic.replace("operator.", "").replace(".request", "")
            if op not in [x[0] for x in seen_ops]:
                seen_ops.append((op, e.iso))

    state["active_agents"] = list(seen_agents)[:10]
    state["recent_operators"] = seen_ops[:5]

    return state


def check_paip_isolation() -> dict:
    """Check PAIP clone isolation status."""
    cwd = Path.cwd()
    is_clone = "/tmp/pluribus_" in str(cwd)

    paip_state = {
        "is_paip_clone": is_clone,
        "working_dir": str(cwd),
        "agent_id": AGENT_ID,
    }

    if is_clone:
        # Extract clone info from path
        parts = str(cwd).split("/")
        for p in parts:
            if p.startswith("pluribus_"):
                paip_state["clone_id"] = p
                break

    return paip_state

# =============================================================================
# Semops Integration
# =============================================================================

def load_semops() -> dict:
    """Load semops registry."""
    try:
        with open(SEMOPS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def get_available_operators() -> List[str]:
    """Get list of available semops operators."""
    semops = load_semops()
    return list(semops.get("operators", {}).keys())


def invoke_operator(operator: str, **kwargs) -> dict:
    """Invoke a semops operator."""
    semops = load_semops()
    op_def = semops.get("operators", {}).get(operator.upper())

    if not op_def:
        return {"ok": False, "error": f"Unknown operator: {operator}"}

    tool_path = Path("/pluribus") / op_def.get("tool", "")
    if not tool_path.exists():
        return {"ok": False, "error": f"Tool not found: {tool_path}"}

    # Build command
    cmd = ["python3", str(tool_path)]
    for k, v in kwargs.items():
        cmd.extend([f"--{k}", str(v)])

    emit_event(f"operator.{operator.lower()}.invoke", {
        "operator": operator,
        "args": kwargs,
    })

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# =============================================================================
# Core Agent Logic
# =============================================================================

def build_system_prompt() -> str:
    """Build the system prompt with current Pluribus context."""
    dkin_state = get_dkin_state()
    paip_state = check_paip_isolation()
    bus_summary = get_bus_summary()

    context = PLURIBUS_SYSTEM_CONTEXT.format(
        agent_id=AGENT_ID,
        protocol_version=PROTOCOL_VERSION,
        paip_version=PAIP_VERSION,
        bus_path=str(BUS_DIR),
    )

    # Add live state
    context += f"""
## Live State
- PBLOCK Active: {dkin_state['pblock_active']}
- Active Agents: {', '.join(dkin_state['active_agents'][:5]) or 'none detected'}
- Recent Operators: {', '.join([x[0] for x in dkin_state['recent_operators']]) or 'none'}
- PAIP Clone: {'Yes - ' + paip_state.get('clone_id', '') if paip_state['is_paip_clone'] else 'No (main tree)'}
- Bus Events (last 200): {bus_summary['event_count']}
"""

    return context


def run_with_context(prompt: str, model: Optional[str] = None, raw: bool = True) -> tuple[int, str]:
    """Run crush with Pluribus system context."""
    req_id = str(uuid.uuid4())[:8]
    session_id = str(uuid.uuid4())[:12]

    # Build full prompt with system context
    system_context = build_system_prompt()
    full_prompt = f"""[SYSTEM CONTEXT]
{system_context}

[USER REQUEST]
{prompt}"""

    emit_event("crush.agent.start", {
        "session_id": session_id,
        "model": model or "default",
        "has_context": True,
    }, req_id=req_id)

    # Build command
    cmd = [CRUSH_BIN]
    if model:
        cmd.extend(["--model", model])
    if raw:
        cmd.append("--raw")
    cmd.append(full_prompt)

    start_time = time.time()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        duration_ms = int((time.time() - start_time) * 1000)

        emit_event("crush.agent.end", {
            "session_id": session_id,
            "ok": proc.returncode == 0,
            "duration_ms": duration_ms,
            "output_length": len(proc.stdout),
        }, req_id=req_id)

        return proc.returncode, proc.stdout

    except Exception as e:
        emit_event("crush.agent.error", {
            "session_id": session_id,
            "error": str(e),
        }, level="error", req_id=req_id)
        return 1, f"Error: {e}"


def interactive_agent(model: Optional[str] = None) -> None:
    """Run interactive agent session with full Pluribus awareness."""
    session_id = str(uuid.uuid4())[:12]

    emit_event("crush.agent.interactive.start", {
        "session_id": session_id,
        "model": model or "default",
    })

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  CRUSH AGENT - Pluribus-Native LLM Interface                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Agent ID: {AGENT_ID:<52} ║
║  Protocol: DKIN {PROTOCOL_VERSION} / PAIP {PAIP_VERSION:<40} ║
║  Bus: {str(BUS_DIR):<56} ║
╠══════════════════════════════════════════════════════════════════╣
║  Commands:                                                        ║
║    /bus          - Show bus summary                               ║
║    /state        - Show DKIN/PAIP state                           ║
║    /operators    - List available operators                       ║
║    /invoke <op>  - Invoke operator (e.g., /invoke ckin)           ║
║    /a2a <target> - Send A2A request                               ║
║    /help         - Show this help                                 ║
║    /quit         - Exit                                           ║
╚══════════════════════════════════════════════════════════════════╝
""")

    turn = 0
    while True:
        try:
            user_input = input("crush> ").strip()
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "/quit":
                    break
                elif cmd == "/help":
                    print("Commands: /bus, /state, /operators, /invoke <op>, /a2a <target>, /quit")
                elif cmd == "/bus":
                    summary = get_bus_summary()
                    print(f"Events: {summary['event_count']}")
                    print(f"Topics: {summary['topics'][:5]}")
                    print(f"Actors: {summary['actors'][:5]}")
                elif cmd == "/state":
                    state = get_dkin_state()
                    print(json.dumps(state, indent=2))
                elif cmd == "/operators":
                    ops = get_available_operators()
                    print(f"Available: {', '.join(ops)}")
                elif cmd == "/invoke":
                    if arg:
                        result = invoke_operator(arg)
                        print(result.get("stdout", result.get("error", "No output")))
                    else:
                        print("Usage: /invoke <operator>")
                elif cmd == "/a2a":
                    if arg:
                        req_id = send_a2a_request(arg, "query", {"from": "interactive"})
                        print(f"A2A request sent: {req_id}")
                    else:
                        print("Usage: /a2a <target_agent>")
                else:
                    print(f"Unknown command: {cmd}")
                continue

            # Regular prompt
            turn += 1
            exit_code, output = run_with_context(user_input, model=model)
            print(output)
            print()

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n[interrupted]")
            break

    emit_event("crush.agent.interactive.end", {
        "session_id": session_id,
        "turns": turn,
    })
    print(f"\n[Session {session_id} ended - {turn} turns]")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Crush Agent - Pluribus-Native LLM Interface"
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to process")
    parser.add_argument("-m", "--model", help="Model to use")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--bus-summary", action="store_true", help="Show bus summary")
    parser.add_argument("--dkin-state", action="store_true", help="Show DKIN state")
    parser.add_argument("--operators", action="store_true", help="List operators")
    parser.add_argument("--invoke", help="Invoke operator")
    parser.add_argument("--a2a-request", help="Send A2A request (JSON)")
    parser.add_argument("--version", action="store_true", help="Show version")

    args = parser.parse_args()

    if args.version:
        print(f"crush_agent {PROTOCOL_VERSION}")
        print(f"Agent ID: {AGENT_ID}")
        print(f"PAIP: {PAIP_VERSION}")
        print(f"A2A: {A2A_VERSION}")
        return

    if args.bus_summary:
        print(json.dumps(get_bus_summary(), indent=2))
        return

    if args.dkin_state:
        print(json.dumps(get_dkin_state(), indent=2))
        return

    if args.operators:
        print("\n".join(get_available_operators()))
        return

    if args.invoke:
        result = invoke_operator(args.invoke)
        print(result.get("stdout", result.get("error", "")))
        return

    if args.a2a_request:
        try:
            req = json.loads(args.a2a_request)
            req_id = send_a2a_request(
                req.get("target", "unknown"),
                req.get("intent", "query"),
                req.get("context", {}),
                req.get("constraints"),
            )
            print(f"A2A request sent: {req_id}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
        return

    if args.interactive:
        interactive_agent(model=args.model)
        return

    if args.prompt:
        exit_code, output = run_with_context(args.prompt, model=args.model)
        print(output)
        sys.exit(exit_code)
    else:
        # Read from stdin if available
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
            if prompt:
                exit_code, output = run_with_context(prompt, model=args.model)
                print(output)
                sys.exit(exit_code)

        parser.print_help()


if __name__ == "__main__":
    main()
