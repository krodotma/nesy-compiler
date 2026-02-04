#!/usr/bin/env python3
"""
PBCMASTER OPERATOR - Codemaster CLI Interface
==============================================

E Pluribus Unum - "From Many, One"

CLI operator for interacting with the Codemaster Agent. All agents must use
this interface to request merges to critical branches (main, staging, dev).

Usage:
    PBCMASTER merge --source <branch> --target <main|staging|dev>
    PBCMASTER status
    PBCMASTER queue
    PBCMASTER rollback --commit <sha> --reason <reason>
    PBCMASTER conserve --source <branch>

Reference: nucleus/specs/codemaster_protocol_v2.md
DKIN Version: v28
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cagent_registry

VERSION = "1.0.0"
PROTOCOL_VERSION = "codemaster-v2"

DEFAULT_BUS_DIR = "/pluribus/.pluribus/bus"
CRITICAL_BRANCHES = {"main", "staging", "dev"}
DEFAULT_CAGENT_REGISTRY = "/pluribus/nucleus/specs/cagent_registry.json"

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_event(bus_dir: Path, topic: str, kind: str, level: str, data: dict, trace_id: Optional[str] = None) -> str:
    """Emit event to bus."""
    import fcntl
    import socket

    events_path = bus_dir / "events.ndjson"
    bus_dir.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())

    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": os.environ.get("PLURIBUS_ACTOR", "pbcmaster"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    if trace_id:
        event["trace_id"] = trace_id

    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"

    with events_path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return event_id


def read_recent_events(bus_dir: Path, topic_prefix: str, limit: int = 100) -> list:
    """Read recent events matching topic prefix."""
    events_path = bus_dir / "events.ndjson"
    if not events_path.exists():
        return []

    events = []
    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if event.get("topic", "").startswith(topic_prefix):
                    events.append(event)
            except json.JSONDecodeError:
                continue

    return events[-limit:]


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


def wait_for_response(bus_dir: Path, trace_id: str, timeout_s: float = 30.0) -> Optional[dict]:
    """Wait for response event with matching trace_id."""
    events_path = bus_dir / "events.ndjson"
    start = time.time()

    seen_positions = set()

    while time.time() - start < timeout_s:
        if events_path.exists():
            with events_path.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i in seen_positions:
                        continue
                    seen_positions.add(i)

                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("trace_id") == trace_id:
                            topic = event.get("topic", "")
                            # Response topics
                            if any(x in topic for x in [".accepted", ".rejected", ".complete", ".failed", ".report"]):
                                return event
                    except json.JSONDecodeError:
                        continue

        time.sleep(0.5)

    return None


def print_box(title: str, content: list, color: str = CYAN) -> None:
    """Print a box with content."""
    width = 70
    print(f"{color}{'═' * width}")
    print(f"{BOLD}{title.center(width)}{RESET}{color}")
    print(f"{'═' * width}{RESET}")
    for line in content:
        print(f"  {line}")
    print(f"{color}{'═' * width}{RESET}")


# =============================================================================
# Commands
# =============================================================================

def cmd_merge(args: argparse.Namespace, bus_dir: Path) -> int:
    """Request a merge to critical branch."""
    source = args.source
    target = args.target

    if target not in CRITICAL_BRANCHES:
        print(f"{RED}Error: Target must be one of: {', '.join(CRITICAL_BRANCHES)}{RESET}")
        return 1

    trace_id = str(uuid.uuid4())

    data = {
        "source_branch": source,
        "target_branch": target,
        "description": args.description or f"Merge {source} into {target}",
        "tests_passed": not args.skip_tests,
        "pbtest_verdict": args.pbtest_verdict,
        "priority": args.priority,
        "conservation_policy": args.conservation_policy,
    }

    citizen_meta = resolve_citizen_meta(os.environ.get("PLURIBUS_ACTOR", "pbcmaster"))
    data.update(
        {
            "citizen_class": citizen_meta.get("citizen_class"),
            "citizen_tier": citizen_meta.get("citizen_tier"),
            "bootstrap_profile": citizen_meta.get("bootstrap_profile"),
            "scope_allowlist": citizen_meta.get("scope_allowlist"),
            "registry_source": citizen_meta.get("registry_source"),
        }
    )

    if args.paip_clone:
        data["paip_clone"] = args.paip_clone

    print(f"{CYAN}Requesting merge: {source} -> {target}{RESET}")

    event_id = emit_event(
        bus_dir,
        topic="codemaster.merge.request",
        kind="request",
        level="info",
        data=data,
        trace_id=trace_id,
    )

    print(f"  Request ID: {event_id[:8]}")
    print(f"  Trace ID: {trace_id[:8]}")

    if args.no_wait:
        print(f"{YELLOW}Not waiting for response (--no-wait){RESET}")
        return 0

    print(f"  Waiting for Codemaster response...")

    response = wait_for_response(bus_dir, trace_id, timeout_s=args.timeout)

    if not response:
        print(f"{YELLOW}Timeout waiting for response. Codemaster may be offline.{RESET}")
        print(f"  Check status with: PBCMASTER status")
        return 1

    topic = response.get("topic", "")
    resp_data = response.get("data", {})

    if ".accepted" in topic:
        print(f"{GREEN}Merge request accepted!{RESET}")
        print(f"  Queue position: {resp_data.get('queue_position', '?')}")
        return 0
    elif ".rejected" in topic:
        print(f"{RED}Merge request rejected!{RESET}")
        print(f"  Reason: {resp_data.get('reason', 'Unknown')}")
        if resp_data.get("conservation_ref"):
            print(f"  Conservation ref: {resp_data.get('conservation_ref')}")
        return 1
    elif ".complete" in topic:
        print(f"{GREEN}Merge complete!{RESET}")
        print(f"  Commit: {resp_data.get('commit_sha', '?')}")
        return 0
    elif ".conflict" in topic:
        print(f"{YELLOW}Merge has conflicts!{RESET}")
        conflicts = resp_data.get("conflicts", [])
        for c in conflicts[:10]:
            print(f"  - {c}")
        if len(conflicts) > 10:
            print(f"  ... and {len(conflicts) - 10} more")
        print(f"  Conservation ref: {resp_data.get('conservation_ref', '?')}")
        return 1
    else:
        print(f"{YELLOW}Unknown response: {topic}{RESET}")
        return 1


def cmd_status(args: argparse.Namespace, bus_dir: Path) -> int:
    """Query Codemaster status."""
    trace_id = str(uuid.uuid4())

    emit_event(
        bus_dir,
        topic="codemaster.status.request",
        kind="request",
        level="info",
        data={},
        trace_id=trace_id,
    )

    print(f"{CYAN}Requesting Codemaster status...{RESET}")

    response = wait_for_response(bus_dir, trace_id, timeout_s=10)

    if not response:
        # Try to get latest health heartbeat
        events = read_recent_events(bus_dir, "codemaster.health.")
        if events:
            last = events[-1]
            data = last.get("data", {})
            age = time.time() - last.get("ts", 0)
            print(f"{YELLOW}Codemaster may be offline. Last heartbeat {age:.0f}s ago.{RESET}")
            print(f"  State: {data.get('state', '?')}")
            print(f"  Queue: {data.get('queue_length', '?')}")
            return 1
        else:
            print(f"{RED}Codemaster appears to be offline.{RESET}")
            print(f"  Start with: python3 nucleus/tools/codemaster_agent.py")
            return 1

    data = response.get("data", {})
    stats = data.get("stats", {})
    branches = data.get("branches", {})

    content = [
        f"State: {BOLD}{data.get('state', '?')}{RESET}",
        f"Uptime: {data.get('uptime_s', 0):.0f}s",
        f"Queue length: {data.get('queue_length', 0)}",
        "",
        f"{BOLD}Statistics:{RESET}",
        f"  Merges: {stats.get('merges_success', 0)} success, {stats.get('merges_conflict', 0)} conflicts, {stats.get('merges_rejected', 0)} rejected",
        f"  Rollbacks: {stats.get('rollbacks', 0)}",
        f"  Conservations: {stats.get('conservations', 0)}",
        "",
        f"{BOLD}Critical Branches:{RESET}",
    ]

    for branch, info in branches.items():
        content.append(f"  {branch}: {info.get('head', '?')[:12]}")

    print_box("CODEMASTER STATUS", content)
    return 0


def cmd_queue(args: argparse.Namespace, bus_dir: Path) -> int:
    """Show merge request queue."""
    events = read_recent_events(bus_dir, "codemaster.merge.")

    # Build queue state
    pending = {}
    completed = []

    for event in events:
        topic = event.get("topic", "")
        data = event.get("data", {})
        req_id = data.get("request_id") or event.get("id")

        if ".request" in topic:
            pending[req_id] = {
                "source": data.get("source_branch", "?"),
                "target": data.get("target_branch", "?"),
                "actor": event.get("actor", "?"),
                "ts": event.get("ts", 0),
                "status": "pending",
            }
        elif ".accepted" in topic:
            if req_id in pending:
                pending[req_id]["status"] = "queued"
        elif ".complete" in topic:
            if req_id in pending:
                pending[req_id]["status"] = "completed"
                completed.append(pending.pop(req_id))
        elif ".rejected" in topic:
            if req_id in pending:
                pending[req_id]["status"] = "rejected"
                pending[req_id]["reason"] = data.get("reason", "?")
        elif ".conflict" in topic:
            if req_id in pending:
                pending[req_id]["status"] = "conflict"

    # Show queue
    content = []

    if pending:
        content.append(f"{BOLD}Pending/Queued:{RESET}")
        for req_id, info in sorted(pending.items(), key=lambda x: x[1]["ts"]):
            age = time.time() - info["ts"]
            status_color = GREEN if info["status"] == "queued" else YELLOW
            content.append(f"  {status_color}{info['status']:10}{RESET} {info['source']} -> {info['target']} ({age:.0f}s ago)")
    else:
        content.append("No pending requests")

    content.append("")
    content.append(f"{BOLD}Recent Completed:{RESET}")
    for info in completed[-5:]:
        content.append(f"  {GREEN}completed{RESET}  {info['source']} -> {info['target']}")

    print_box("MERGE QUEUE", content)
    return 0


def cmd_rollback(args: argparse.Namespace, bus_dir: Path) -> int:
    """Request a rollback."""
    trace_id = str(uuid.uuid4())

    print(f"{YELLOW}Requesting rollback: {args.branch} -> {args.commit}{RESET}")
    print(f"  Reason: {args.reason}")

    if not args.force:
        confirm = input(f"{RED}This is a destructive operation. Continue? [y/N] {RESET}")
        if confirm.lower() != "y":
            print("Aborted.")
            return 1

    emit_event(
        bus_dir,
        topic="codemaster.rollback.request",
        kind="request",
        level="warn",
        data={
            "commit": args.commit,
            "branch": args.branch,
            "reason": args.reason,
        },
        trace_id=trace_id,
    )

    response = wait_for_response(bus_dir, trace_id, timeout_s=60)

    if not response:
        print(f"{RED}Timeout waiting for rollback response.{RESET}")
        return 1

    topic = response.get("topic", "")
    data = response.get("data", {})

    if ".complete" in topic:
        print(f"{GREEN}Rollback complete!{RESET}")
        print(f"  From: {data.get('from_commit', '?')}")
        print(f"  To: {data.get('to_commit', '?')}")
        return 0
    else:
        print(f"{RED}Rollback failed!{RESET}")
        print(f"  Error: {data.get('error', 'Unknown')}")
        return 1


def cmd_conserve(args: argparse.Namespace, bus_dir: Path) -> int:
    """Request conservation of a branch."""
    trace_id = str(uuid.uuid4())

    print(f"{CYAN}Requesting conservation: {args.source}{RESET}")
    print(f"  Description: {args.description}")

    emit_event(
        bus_dir,
        topic="codemaster.conservation.request",
        kind="request",
        level="info",
        data={
            "source": args.source,
            "description": args.description,
        },
        trace_id=trace_id,
    )

    response = wait_for_response(bus_dir, trace_id, timeout_s=30)

    if not response:
        print(f"{YELLOW}Timeout waiting for response. Conservation may still succeed.{RESET}")
        return 1

    topic = response.get("topic", "")
    data = response.get("data", {})

    if ".complete" in topic:
        print(f"{GREEN}Conservation complete!{RESET}")
        print(f"  Preserved at: {data.get('preserved_at', '?')}")
        return 0
    else:
        print(f"{RED}Conservation failed!{RESET}")
        print(f"  Reason: {data.get('reason', 'Unknown')}")
        return 1


def cmd_report(args: argparse.Namespace, bus_dir: Path) -> int:
    """Generate PBREPORT-style status report."""
    events = read_recent_events(bus_dir, "codemaster.", limit=500)

    # Calculate stats
    today = time.strftime("%Y-%m-%d")
    today_merges = 0
    today_conflicts = 0
    today_rejected = 0
    today_conservations = 0

    actor_stats = {}

    for event in events:
        ts = event.get("ts", 0)
        event_date = time.strftime("%Y-%m-%d", time.gmtime(ts))
        topic = event.get("topic", "")
        actor = event.get("actor", "unknown")

        if event_date == today:
            if ".merge.complete" in topic:
                today_merges += 1
                actor_stats.setdefault(actor, {"merges": 0, "conflicts": 0})
                actor_stats[actor]["merges"] += 1
            elif ".merge.conflict" in topic:
                today_conflicts += 1
                actor_stats.setdefault(actor, {"merges": 0, "conflicts": 0})
                actor_stats[actor]["conflicts"] += 1
            elif ".merge.rejected" in topic:
                today_rejected += 1
            elif ".conservation.complete" in topic:
                today_conservations += 1

    # Get latest status
    status_events = [e for e in events if e.get("topic") == "codemaster.health.heartbeat"]
    if status_events:
        latest = status_events[-1]
        state = latest.get("data", {}).get("state", "unknown")
        uptime = latest.get("data", {}).get("uptime_s", 0)
    else:
        state = "offline"
        uptime = 0

    # Format report
    lines = [
        f"╔{'═' * 69}╗",
        f"║{'CODEMASTER STATUS REPORT'.center(69)}║",
        f"║{now_iso().center(69)}║",
        f"╠{'═' * 69}╣",
        f"║ STATE: {state.upper():<61}║",
        f"║ UPTIME: {uptime:.0f}s{' ' * (60 - len(str(int(uptime))))}║",
        f"╠{'═' * 69}╣",
        f"║ TODAY'S ACTIVITY{' ' * 52}║",
        f"║ ├── Merges:        {today_merges:<47}║",
        f"║ ├── Conflicts:     {today_conflicts:<47}║",
        f"║ ├── Rejected:      {today_rejected:<47}║",
        f"║ └── Conserved:     {today_conservations:<47}║",
        f"╠{'═' * 69}╣",
        f"║ AGENT CONTRIBUTIONS (today){' ' * 40}║",
    ]

    for actor, stats in sorted(actor_stats.items()):
        m = stats["merges"]
        c = stats["conflicts"]
        line = f"║ ├── {actor[:15]:<15}: {m} merges, {c} conflicts"
        lines.append(f"{line:<70}║")

    if not actor_stats:
        lines.append(f"║ ├── (no activity){' ' * 50}║")

    lines.append(f"╚{'═' * 69}╝")

    print("\n".join(lines))
    return 0


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PBCMASTER - Codemaster CLI Operator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {VERSION}
Protocol: {PROTOCOL_VERSION}

E Pluribus Unum - "From Many, One"
All merges to critical branches (main, staging, dev) must go through Codemaster.

Examples:
    PBCMASTER merge --source feature/foo --target main
    PBCMASTER merge --source bugfix/bar --target staging --priority 8
    PBCMASTER status
    PBCMASTER queue
    PBCMASTER rollback --commit abc123 --reason "Test regression" --force
    PBCMASTER conserve --source orphan-branch --description "Orphan from cleanup"

Environment:
    PLURIBUS_BUS_DIR    Bus directory (default: {DEFAULT_BUS_DIR})
    PLURIBUS_ACTOR      Actor name for bus events

Reference: nucleus/specs/codemaster_protocol_v2.md
""",
    )

    parser.add_argument(
        "--bus-dir",
        default=os.environ.get("PLURIBUS_BUS_DIR", DEFAULT_BUS_DIR),
        help=f"Bus directory (default: {DEFAULT_BUS_DIR})",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"pbcmaster {VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # merge command
    merge_parser = subparsers.add_parser("merge", help="Request merge to critical branch")
    merge_parser.add_argument("--source", "-s", required=True, help="Source branch")
    merge_parser.add_argument("--target", "-t", required=True, help="Target branch (main/staging/dev)")
    merge_parser.add_argument("--description", "-d", help="Merge description")
    merge_parser.add_argument("--priority", "-p", type=int, default=5, help="Priority 1-10 (default: 5)")
    merge_parser.add_argument("--pbtest-verdict", default="pass", choices=["pass", "fail", "skip"])
    merge_parser.add_argument("--skip-tests", action="store_true", help="Mark tests as not passed")
    merge_parser.add_argument("--paip-clone", help="PAIP clone path if applicable")
    merge_parser.add_argument("--conservation-policy", default="always", choices=["always", "on_conflict", "never"])
    merge_parser.add_argument("--no-wait", action="store_true", help="Don't wait for response")
    merge_parser.add_argument("--timeout", type=float, default=30.0, help="Response timeout in seconds")

    # status command
    subparsers.add_parser("status", help="Query Codemaster status")

    # queue command
    subparsers.add_parser("queue", help="Show merge request queue")

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Request rollback")
    rollback_parser.add_argument("--commit", "-c", required=True, help="Commit SHA to rollback to")
    rollback_parser.add_argument("--branch", "-b", default="main", help="Branch to rollback (default: main)")
    rollback_parser.add_argument("--reason", "-r", required=True, help="Reason for rollback")
    rollback_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # conserve command
    conserve_parser = subparsers.add_parser("conserve", help="Request conservation")
    conserve_parser.add_argument("--source", "-s", required=True, help="Branch/ref to conserve")
    conserve_parser.add_argument("--description", "-d", default="Manual conservation", help="Description")

    # report command
    subparsers.add_parser("report", help="Generate PBREPORT-style status report")

    args = parser.parse_args()
    bus_dir = Path(args.bus_dir)

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "merge": cmd_merge,
        "status": cmd_status,
        "queue": cmd_queue,
        "rollback": cmd_rollback,
        "conserve": cmd_conserve,
        "report": cmd_report,
    }

    return commands[args.command](args, bus_dir)


if __name__ == "__main__":
    sys.exit(main())
