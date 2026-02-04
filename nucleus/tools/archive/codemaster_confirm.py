#!/usr/bin/env python3
"""
CODEMASTER RULES CONFIRMATION
=============================

Script for agents to:
1. Check for Codemaster rules broadcast
2. Confirm receipt and understanding
3. View which agents have confirmed

Usage:
    python3 codemaster_confirm.py check      # View the rules
    python3 codemaster_confirm.py confirm    # Confirm as current agent
    python3 codemaster_confirm.py status     # See who has confirmed
"""
import json
import os
import socket
import sys
import time
import uuid
import fcntl
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cagent_registry

BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
EVENTS_PATH = BUS_DIR / "events.ndjson"
DEFAULT_CAGENT_REGISTRY = "/pluribus/nucleus/specs/cagent_registry.json"

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def read_events(topic_prefix: str) -> list:
    """Read events matching topic prefix."""
    if not EVENTS_PATH.exists():
        return []
    events = []
    with EVENTS_PATH.open("r", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if event.get("topic", "").startswith(topic_prefix):
                    events.append(event)
            except json.JSONDecodeError:
                continue
    return events


def resolve_citizen_meta(actor: str) -> dict:
    registry_path = Path(os.environ.get("PLURIBUS_CAGENT_REGISTRY", DEFAULT_CAGENT_REGISTRY))
    try:
        registry = cagent_registry.load_registry(registry_path)
    except Exception:
        registry = {
            "defaults": {
                "citizen_class": "superworker",
                "citizen_tier": "limited",
                "bootstrap_profile": "minimal",
                "scope_allowlist": [],
            },
            "actors": [],
            "class_aliases": {},
            "tier_aliases": {},
        }
    overrides = cagent_registry.env_overrides()
    profile = cagent_registry.resolve_actor(actor, registry, overrides=overrides, allow_override=True)
    return {
        "citizen_class": profile.citizen_class,
        "citizen_tier": profile.citizen_tier,
        "bootstrap_profile": profile.bootstrap_profile,
        "scope_allowlist": profile.scope_allowlist,
        "registry_source": profile.source,
    }


def emit_event(topic: str, kind: str, level: str, data: dict) -> str:
    """Emit event to bus."""
    BUS_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": os.environ.get("PLURIBUS_ACTOR", os.environ.get("USER", "unknown")),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }
    line = json.dumps(event, separators=(",", ":")) + "\n"
    with EVENTS_PATH.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return event["id"]


def cmd_check():
    """Display the Codemaster rules broadcast."""
    broadcasts = read_events("codemaster.rules.broadcast")

    if not broadcasts:
        print(f"{RED}No Codemaster rules broadcast found on bus.{RESET}")
        return 1

    latest = broadcasts[-1]
    data = latest.get("data", {})

    print(f"\n{CYAN}{'═' * 70}")
    print(f"{BOLD}CODEMASTER RULES BROADCAST{RESET}{CYAN}")
    print(f"{'═' * 70}{RESET}\n")

    print(f"{BOLD}Broadcast ID:{RESET} {latest['id'][:8]}")
    print(f"{BOLD}Timestamp:{RESET} {latest.get('iso', '?')}")
    print(f"{BOLD}Version:{RESET} {data.get('version', '?')}")
    print(f"{BOLD}Protocol:{RESET} {data.get('protocol', '?')}")
    print(f"{BOLD}Commit:{RESET} {data.get('commit', '?')}")
    print(f"\n{BOLD}Philosophy:{RESET} {data.get('philosophy', '?')}")

    print(f"\n{YELLOW}{BOLD}MANDATORY RULES:{RESET}")
    for i, rule in enumerate(data.get("mandatory_rules", []), 1):
        print(f"  {i}. {rule}")

    print(f"\n{BOLD}Critical Branches (require Codemaster):{RESET}")
    for branch in data.get("critical_branches", []):
        print(f"  - {RED}{branch}{RESET}")

    print(f"\n{BOLD}Agent-Owned Paths (no Codemaster needed):{RESET}")
    for path in data.get("agent_owned_paths", []):
        print(f"  - {GREEN}{path}{RESET}")

    print(f"\n{BOLD}CLI Commands:{RESET}")
    for cmd, usage in data.get("cli_commands", {}).items():
        print(f"  {cmd}: {usage}")

    print(f"\n{BOLD}Conservation:{RESET} {data.get('conservation_policy', '?')}")

    print(f"\n{BOLD}Documentation:{RESET}")
    for doc in data.get("documentation", []):
        print(f"  - {doc}")

    print(f"\n{CYAN}{'═' * 70}{RESET}")
    print(f"\n{YELLOW}To confirm receipt, run:{RESET}")
    print(f"  python3 nucleus/tools/codemaster_confirm.py confirm")
    print()

    return 0


def cmd_confirm():
    """Confirm receipt of Codemaster rules."""
    broadcasts = read_events("codemaster.rules.broadcast")

    if not broadcasts:
        print(f"{RED}No Codemaster rules broadcast found. Cannot confirm.{RESET}")
        return 1

    latest = broadcasts[-1]
    broadcast_id = latest["id"]
    data = latest.get("data", {})

    actor = os.environ.get("PLURIBUS_ACTOR", os.environ.get("USER", "unknown"))
    citizen_meta = resolve_citizen_meta(actor)

    # Check if already confirmed
    confirmations = read_events("codemaster.rules.confirmed")
    for conf in confirmations:
        if conf.get("actor") == actor and conf.get("data", {}).get("broadcast_id") == broadcast_id:
            print(f"{YELLOW}Agent '{actor}' has already confirmed this broadcast.{RESET}")
            return 0

    # Emit confirmation
    event_id = emit_event(
        topic="codemaster.rules.confirmed",
        kind="response",
        level="info",
        data={
            "broadcast_id": broadcast_id,
            "acknowledged": True,
            "rules_version": data.get("version", "unknown"),
            "understood_rules": [
                "no direct push to critical branches",
                "use PBCMASTER for merges",
                "include PBTEST verdict",
                "trust Codemaster for conflicts"
            ],
            "citizen_meta": citizen_meta,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )

    print(f"{GREEN}Codemaster rules confirmed!{RESET}")
    print(f"  Agent: {actor}")
    print(f"  Broadcast ID: {broadcast_id[:8]}")
    print(f"  Confirmation ID: {event_id[:8]}")
    print(f"  Version: {data.get('version', '?')}")

    return 0


def cmd_status():
    """Show confirmation status."""
    broadcasts = read_events("codemaster.rules.broadcast")
    confirmations = read_events("codemaster.rules.confirmed")

    if not broadcasts:
        print(f"{RED}No Codemaster rules broadcast found.{RESET}")
        return 1

    latest = broadcasts[-1]
    broadcast_id = latest["id"]
    broadcast_ts = latest.get("ts", 0)

    print(f"\n{CYAN}{'═' * 70}")
    print(f"{BOLD}CODEMASTER RULES CONFIRMATION STATUS{RESET}{CYAN}")
    print(f"{'═' * 70}{RESET}\n")

    print(f"{BOLD}Broadcast:{RESET} {broadcast_id[:8]}")
    print(f"{BOLD}Time:{RESET} {latest.get('iso', '?')}")
    print(f"{BOLD}Version:{RESET} {latest.get('data', {}).get('version', '?')}")

    # Filter confirmations for this broadcast
    relevant = [c for c in confirmations if c.get("data", {}).get("broadcast_id") == broadcast_id]

    print(f"\n{BOLD}Confirmations ({len(relevant)}):{RESET}")

    if relevant:
        confirmed_agents = set()
        for conf in relevant:
            actor = conf.get("actor", "?")
            if actor in confirmed_agents:
                continue
            confirmed_agents.add(actor)
            conf_ts = conf.get("ts", 0)
            delay = conf_ts - broadcast_ts
            print(f"  {GREEN}✓{RESET} {actor:<20} confirmed at {conf.get('iso', '?')} (+{delay:.0f}s)")
    else:
        print(f"  {YELLOW}No confirmations yet.{RESET}")

    # Known agents that should confirm
    known_agents = {"claude", "codex", "gemini", "qwen", "aider", "omega"}
    confirmed_agents = {c.get("actor") for c in relevant}
    missing = known_agents - confirmed_agents

    if missing:
        print(f"\n{BOLD}Awaiting confirmation:{RESET}")
        for agent in sorted(missing):
            print(f"  {RED}○{RESET} {agent}")

    print(f"\n{CYAN}{'═' * 70}{RESET}\n")

    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1].lower()

    if cmd == "check":
        return cmd_check()
    elif cmd == "confirm":
        return cmd_confirm()
    elif cmd == "status":
        return cmd_status()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
