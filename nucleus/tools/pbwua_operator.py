#!/usr/bin/env python3
"""
PBWUA OPERATOR - Pluribus Web User Agent CLI
=============================================

CLI for managing browser automation sessions that drive webchat interfaces.

Usage:
    PBWUA status                    # Show all sessions
    PBWUA start <provider>          # Start session for provider
    PBWUA stop <provider>           # Stop session gracefully
    PBWUA auth <provider>           # Check auth state
    PBWUA send <provider> <message> # Send message via bus
    PBWUA screenshot <provider>     # Capture current screen
    PBWUA health                    # Overall WUA health check
    PBWUA logs [--tail N]           # Show recent WUA bus events

Reference: nucleus/specs/wua_protocol_v1.md
DKIN Version: v29
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True

VERSION = "1.0.0"
PROTOCOL_VERSION = "wua-v1"

DEFAULT_BUS_DIR = "/pluribus/.pluribus/bus"
ACTOR = "pbwua-operator"

PROVIDERS = ["chatgpt", "claude", "gemini"]

# ANSI colors - only use if terminal supports them
def supports_color() -> bool:
    """Check if terminal supports color output."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return True


USE_COLOR = supports_color()

GREEN = "\033[92m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
RED = "\033[91m" if USE_COLOR else ""
CYAN = "\033[96m" if USE_COLOR else ""
BLUE = "\033[94m" if USE_COLOR else ""
MAGENTA = "\033[95m" if USE_COLOR else ""
BOLD = "\033[1m" if USE_COLOR else ""
DIM = "\033[2m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""


def now_iso() -> str:
    """Return current time in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_event(bus_dir: Path, topic: str, kind: str, level: str, data: dict, trace_id: Optional[str] = None) -> str:
    """Emit event to bus."""
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
        "actor": os.environ.get("PLURIBUS_ACTOR", ACTOR),
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


def read_recent_events(bus_dir: Path, topic_prefix: str, limit: int = 50) -> list:
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


def wait_for_response(bus_dir: Path, trace_id: str, topic_contains: str, timeout_s: float = 30.0) -> Optional[dict]:
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
                            if topic_contains in topic:
                                return event
                    except json.JSONDecodeError:
                        continue

        time.sleep(0.3)

    return None


def format_age(seconds: float) -> str:
    """Format age in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def print_box(title: str, content: list, color: str = CYAN, width: int = 72) -> None:
    """Print a formatted box with title and content."""
    print(f"{color}{'=' * width}")
    print(f"{BOLD}{title.center(width)}{RESET}{color}")
    print(f"{'=' * width}{RESET}")
    for line in content:
        print(f"  {line}")
    print(f"{color}{'=' * width}{RESET}")


def provider_color(provider: str) -> str:
    """Get color for provider."""
    colors = {
        "chatgpt": GREEN,
        "claude": MAGENTA,
        "gemini": BLUE,
    }
    return colors.get(provider, CYAN)


def status_indicator(status: str) -> str:
    """Get status indicator with color."""
    indicators = {
        "active": f"{GREEN}[ACTIVE]{RESET}",
        "idle": f"{CYAN}[IDLE]{RESET}",
        "starting": f"{YELLOW}[STARTING]{RESET}",
        "stopping": f"{YELLOW}[STOPPING]{RESET}",
        "error": f"{RED}[ERROR]{RESET}",
        "offline": f"{DIM}[OFFLINE]{RESET}",
        "auth_required": f"{YELLOW}[AUTH REQUIRED]{RESET}",
        "authenticated": f"{GREEN}[AUTHENTICATED]{RESET}",
    }
    return indicators.get(status.lower(), f"[{status.upper()}]")


# =============================================================================
# Commands
# =============================================================================

def cmd_status(args: argparse.Namespace, bus_dir: Path) -> int:
    """Show status of all WUA sessions."""
    # Gather status for each provider
    sessions = {}

    for provider in PROVIDERS:
        # Look for recent status events
        events = read_recent_events(bus_dir, f"wua.session.{provider}", limit=20)

        # Also check heartbeats
        heartbeats = read_recent_events(bus_dir, f"wua.heartbeat.{provider}", limit=5)

        if events or heartbeats:
            # Get latest info
            latest = None
            if events:
                latest = events[-1]
            if heartbeats and (not latest or heartbeats[-1].get("ts", 0) > latest.get("ts", 0)):
                latest = heartbeats[-1]

            if latest:
                data = latest.get("data", {})
                sessions[provider] = {
                    "status": data.get("status", "unknown"),
                    "confidence": data.get("confidence", "?"),
                    "last_seen": latest.get("ts", 0),
                    "auth_state": data.get("auth_state", "unknown"),
                    "messages_sent": data.get("messages_sent", 0),
                    "messages_received": data.get("messages_received", 0),
                }
        else:
            sessions[provider] = {
                "status": "offline",
                "confidence": "-",
                "last_seen": 0,
                "auth_state": "unknown",
                "messages_sent": 0,
                "messages_received": 0,
            }

    # Format output
    content = [
        f"{BOLD}{'PROVIDER':<12} {'STATUS':<16} {'AUTH':<16} {'LAST SEEN':<12} {'MSGS'}{RESET}",
        "-" * 68,
    ]

    for provider in PROVIDERS:
        info = sessions[provider]
        pcolor = provider_color(provider)
        status = status_indicator(info["status"])

        if info["last_seen"] > 0:
            age = format_age(time.time() - info["last_seen"])
        else:
            age = "-"

        auth = info["auth_state"]
        if auth == "authenticated":
            auth_disp = f"{GREEN}OK{RESET}"
        elif auth == "required":
            auth_disp = f"{YELLOW}NEEDED{RESET}"
        else:
            auth_disp = f"{DIM}{auth}{RESET}"

        msgs = f"{info['messages_sent']}/{info['messages_received']}"

        content.append(
            f"{pcolor}{provider:<12}{RESET} {status:<25} {auth_disp:<25} {age:<12} {msgs}"
        )

    content.append("")
    content.append(f"{DIM}MSGS = sent/received{RESET}")

    print_box("WUA SESSION STATUS", content)
    return 0


def cmd_start(args: argparse.Namespace, bus_dir: Path) -> int:
    """Request session start for provider."""
    provider = args.provider
    trace_id = str(uuid.uuid4())

    print(f"{CYAN}Starting WUA session for {provider}...{RESET}")

    data = {
        "provider": provider,
        "headless": args.headless,
        "profile": args.profile,
    }

    if args.url:
        data["url"] = args.url

    emit_event(
        bus_dir,
        topic=f"wua.control.start",
        kind="command",
        level="info",
        data=data,
        trace_id=trace_id,
    )

    print(f"  Trace ID: {trace_id[:8]}...")

    if args.no_wait:
        print(f"{YELLOW}Not waiting for response (--no-wait){RESET}")
        return 0

    print(f"  Waiting for session to start...")

    response = wait_for_response(bus_dir, trace_id, "wua.session", timeout_s=args.timeout)

    if not response:
        print(f"{YELLOW}Timeout waiting for session start.{RESET}")
        print(f"  The session may still be starting. Check with: PBWUA status")
        return 1

    topic = response.get("topic", "")
    data = response.get("data", {})

    if ".started" in topic or data.get("status") == "active":
        print(f"{GREEN}Session started successfully!{RESET}")
        print(f"  Provider: {provider}")
        print(f"  Status: {data.get('status', 'unknown')}")
        if data.get("pid"):
            print(f"  PID: {data.get('pid')}")
        return 0
    elif ".error" in topic or ".failed" in topic:
        print(f"{RED}Session start failed!{RESET}")
        print(f"  Error: {data.get('error', 'Unknown error')}")
        return 1
    else:
        print(f"{YELLOW}Unexpected response: {topic}{RESET}")
        return 1


def cmd_stop(args: argparse.Namespace, bus_dir: Path) -> int:
    """Request graceful session stop."""
    provider = args.provider
    trace_id = str(uuid.uuid4())

    print(f"{YELLOW}Stopping WUA session for {provider}...{RESET}")

    emit_event(
        bus_dir,
        topic=f"wua.control.stop",
        kind="command",
        level="info",
        data={
            "provider": provider,
            "graceful": not args.force,
        },
        trace_id=trace_id,
    )

    print(f"  Trace ID: {trace_id[:8]}...")

    if args.no_wait:
        print(f"{YELLOW}Not waiting for response (--no-wait){RESET}")
        return 0

    response = wait_for_response(bus_dir, trace_id, "wua.session", timeout_s=15)

    if not response:
        print(f"{YELLOW}No response received. Session may already be stopped.{RESET}")
        return 0

    data = response.get("data", {})
    if data.get("status") in ["stopped", "offline"]:
        print(f"{GREEN}Session stopped.{RESET}")
        return 0
    else:
        print(f"{YELLOW}Session status: {data.get('status', 'unknown')}{RESET}")
        return 0


def cmd_auth(args: argparse.Namespace, bus_dir: Path) -> int:
    """Check authentication state."""
    provider = args.provider
    trace_id = str(uuid.uuid4())

    print(f"{CYAN}Checking auth state for {provider}...{RESET}")

    emit_event(
        bus_dir,
        topic=f"wua.session.probe",
        kind="request",
        level="info",
        data={
            "provider": provider,
            "check": "auth",
        },
        trace_id=trace_id,
    )

    response = wait_for_response(bus_dir, trace_id, "wua.", timeout_s=10)

    if not response:
        # Try to find latest auth state from status events
        events = read_recent_events(bus_dir, f"wua.session.{provider}", limit=10)
        if events:
            latest = events[-1]
            data = latest.get("data", {})
            auth = data.get("auth_state", "unknown")
            age = format_age(time.time() - latest.get("ts", 0))

            print(f"  Auth state: {auth} (as of {age} ago)")
            if auth == "authenticated":
                print(f"{GREEN}  Session is authenticated.{RESET}")
                return 0
            elif auth == "required":
                print(f"{YELLOW}  Authentication required. Please log in manually.{RESET}")
                return 1
            else:
                print(f"{DIM}  Auth state unknown.{RESET}")
                return 1
        else:
            print(f"{YELLOW}No session information available for {provider}.{RESET}")
            print(f"  Start a session with: PBWUA start {provider}")
            return 1

    data = response.get("data", {})
    auth = data.get("auth_state", "unknown")

    if auth == "authenticated":
        print(f"{GREEN}Authenticated as: {data.get('username', 'unknown')}{RESET}")
        return 0
    elif auth == "required":
        print(f"{YELLOW}Authentication required.{RESET}")
        print(f"  Please log in to {provider} in the browser window.")
        return 1
    else:
        print(f"{DIM}Auth state: {auth}{RESET}")
        return 1


def cmd_send(args: argparse.Namespace, bus_dir: Path) -> int:
    """Send message to provider via bus."""
    provider = args.provider
    message = " ".join(args.message)
    trace_id = str(uuid.uuid4())

    print(f"{CYAN}Sending message to {provider}...{RESET}")
    print(f"  Message: {message[:50]}{'...' if len(message) > 50 else ''}")

    emit_event(
        bus_dir,
        topic=f"wua.input.inject",
        kind="command",
        level="info",
        data={
            "provider": provider,
            "message": message,
            "await_response": not args.no_response,
            "timeout_s": args.timeout,
        },
        trace_id=trace_id,
    )

    print(f"  Trace ID: {trace_id[:8]}...")

    if args.no_response:
        print(f"{GREEN}Message sent. Not waiting for response.{RESET}")
        return 0

    print(f"  Waiting for response...")

    response = wait_for_response(bus_dir, trace_id, "wua.output", timeout_s=args.timeout)

    if not response:
        print(f"{YELLOW}Timeout waiting for response.{RESET}")
        print(f"  The message may still be processing. Check logs with: PBWUA logs")
        return 1

    data = response.get("data", {})
    output = data.get("response", data.get("text", ""))
    confidence = data.get("confidence", "?")

    print(f"\n{BOLD}Response:{RESET}")
    print("-" * 60)
    print(output)
    print("-" * 60)
    print(f"{DIM}Confidence: {confidence}{RESET}")

    return 0


def cmd_screenshot(args: argparse.Namespace, bus_dir: Path) -> int:
    """Request screenshot capture."""
    provider = args.provider
    trace_id = str(uuid.uuid4())

    print(f"{CYAN}Capturing screenshot for {provider}...{RESET}")

    emit_event(
        bus_dir,
        topic=f"wua.control.screenshot",
        kind="command",
        level="info",
        data={
            "provider": provider,
            "output_path": args.output,
        },
        trace_id=trace_id,
    )

    response = wait_for_response(bus_dir, trace_id, "wua.screenshot", timeout_s=15)

    if not response:
        print(f"{YELLOW}Timeout waiting for screenshot.{RESET}")
        return 1

    data = response.get("data", {})
    path = data.get("path")

    if path:
        print(f"{GREEN}Screenshot saved: {path}{RESET}")
        return 0
    elif data.get("error"):
        print(f"{RED}Screenshot failed: {data.get('error')}{RESET}")
        return 1
    else:
        print(f"{YELLOW}Screenshot response received but no path provided.{RESET}")
        return 1


def cmd_health(args: argparse.Namespace, bus_dir: Path) -> int:
    """Overall health check."""
    print(f"{CYAN}WUA Health Check{RESET}")
    print("=" * 60)

    healthy_count = 0
    unhealthy_count = 0

    for provider in PROVIDERS:
        # Check for recent heartbeats
        heartbeats = read_recent_events(bus_dir, f"wua.heartbeat.{provider}", limit=5)
        status_events = read_recent_events(bus_dir, f"wua.session.{provider}", limit=5)

        all_events = heartbeats + status_events
        if all_events:
            all_events.sort(key=lambda e: e.get("ts", 0))
            latest = all_events[-1]
            age = time.time() - latest.get("ts", 0)

            # Consider healthy if seen within last 5 minutes
            if age < 300:
                status = f"{GREEN}HEALTHY{RESET}"
                healthy_count += 1
            elif age < 900:
                status = f"{YELLOW}STALE{RESET}"
                unhealthy_count += 1
            else:
                status = f"{RED}OFFLINE{RESET}"
                unhealthy_count += 1

            print(f"  {provider_color(provider)}{provider:<12}{RESET} {status} (last seen {format_age(age)} ago)")
        else:
            print(f"  {provider_color(provider)}{provider:<12}{RESET} {DIM}NO DATA{RESET}")
            unhealthy_count += 1

    print("=" * 60)

    # Check bus health
    events_path = bus_dir / "events.ndjson"
    if events_path.exists():
        bus_size = events_path.stat().st_size
        bus_status = f"{GREEN}OK{RESET}" if bus_size < 100 * 1024 * 1024 else f"{YELLOW}LARGE{RESET}"
        print(f"  Bus file: {bus_status} ({bus_size / 1024:.1f} KB)")
    else:
        print(f"  Bus file: {RED}NOT FOUND{RESET}")

    print("=" * 60)

    if unhealthy_count == 0:
        print(f"{GREEN}All systems healthy.{RESET}")
        return 0
    elif healthy_count > 0:
        print(f"{YELLOW}{healthy_count} healthy, {unhealthy_count} issues.{RESET}")
        return 0
    else:
        print(f"{RED}No healthy sessions.{RESET}")
        return 1


def cmd_logs(args: argparse.Namespace, bus_dir: Path) -> int:
    """Show recent WUA bus events."""
    events = read_recent_events(bus_dir, "wua.", limit=args.tail)

    if not events:
        print(f"{YELLOW}No WUA events found.{RESET}")
        return 0

    print(f"{BOLD}Recent WUA Events (last {len(events)}):{RESET}")
    print("-" * 80)

    for event in events:
        ts = event.get("ts", 0)
        iso = time.strftime("%H:%M:%S", time.localtime(ts))
        topic = event.get("topic", "?")
        level = event.get("level", "info")
        data = event.get("data", {})

        # Color by level
        if level == "error":
            lcolor = RED
        elif level == "warn":
            lcolor = YELLOW
        else:
            lcolor = DIM

        # Extract key info
        summary = ""
        if "provider" in data:
            summary = f"[{data['provider']}]"
        if "status" in data:
            summary += f" status={data['status']}"
        if "message" in data:
            msg = data["message"]
            summary += f" msg={msg[:30]}..." if len(str(msg)) > 30 else f" msg={msg}"
        if "error" in data:
            summary += f" error={data['error']}"

        # Shorten topic for display
        short_topic = topic.replace("wua.", "")

        print(f"{lcolor}{iso}{RESET} {CYAN}{short_topic:<25}{RESET} {summary}")

    print("-" * 80)
    return 0


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PBWUA - Pluribus Web User Agent Operator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {VERSION}
Protocol: {PROTOCOL_VERSION}

Manage browser automation sessions for webchat interfaces.

Commands:
    status              Show status of all WUA sessions
    start <provider>    Start a new browser session
    stop <provider>     Stop a browser session gracefully
    auth <provider>     Check authentication state
    send <provider>     Send a message and get response
    screenshot          Capture current browser screen
    health              Overall WUA health check
    logs                Show recent WUA bus events

Providers: {', '.join(PROVIDERS)}

Examples:
    PBWUA status
    PBWUA start chatgpt
    PBWUA auth claude
    PBWUA send gemini "What is 2+2?"
    PBWUA screenshot chatgpt --output /tmp/screen.png
    PBWUA logs --tail 50

Environment:
    PLURIBUS_BUS_DIR    Bus directory (default: {DEFAULT_BUS_DIR})
    PLURIBUS_ACTOR      Actor name for bus events
    NO_COLOR            Disable colored output

Reference: nucleus/specs/wua_protocol_v1.md
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
        version=f"pbwua {VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status command
    subparsers.add_parser("status", help="Show all WUA sessions")

    # start command
    p_start = subparsers.add_parser("start", help="Start a browser session")
    p_start.add_argument("provider", choices=PROVIDERS, help="Provider to start")
    p_start.add_argument("--headless", action="store_true", help="Run in headless mode")
    p_start.add_argument("--profile", help="Browser profile to use")
    p_start.add_argument("--url", help="Custom URL to navigate to")
    p_start.add_argument("--no-wait", action="store_true", help="Don't wait for response")
    p_start.add_argument("--timeout", type=float, default=60.0, help="Timeout in seconds (default: 60)")

    # stop command
    p_stop = subparsers.add_parser("stop", help="Stop a browser session")
    p_stop.add_argument("provider", choices=PROVIDERS, help="Provider to stop")
    p_stop.add_argument("--force", "-f", action="store_true", help="Force stop without graceful shutdown")
    p_stop.add_argument("--no-wait", action="store_true", help="Don't wait for response")

    # auth command
    p_auth = subparsers.add_parser("auth", help="Check authentication state")
    p_auth.add_argument("provider", choices=PROVIDERS, help="Provider to check")

    # send command
    p_send = subparsers.add_parser("send", help="Send message and get response")
    p_send.add_argument("provider", choices=PROVIDERS, help="Provider to send to")
    p_send.add_argument("message", nargs="+", help="Message to send")
    p_send.add_argument("--no-response", action="store_true", help="Don't wait for response")
    p_send.add_argument("--timeout", type=float, default=120.0, help="Response timeout in seconds (default: 120)")

    # screenshot command
    p_ss = subparsers.add_parser("screenshot", help="Capture browser screenshot")
    p_ss.add_argument("provider", choices=PROVIDERS, help="Provider to capture")
    p_ss.add_argument("--output", "-o", help="Output path for screenshot")

    # health command
    subparsers.add_parser("health", help="Overall WUA health check")

    # logs command
    p_logs = subparsers.add_parser("logs", help="Show recent WUA bus events")
    p_logs.add_argument("--tail", type=int, default=20, help="Number of events to show (default: 20)")

    args = parser.parse_args()
    bus_dir = Path(args.bus_dir)

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "status": cmd_status,
        "start": cmd_start,
        "stop": cmd_stop,
        "auth": cmd_auth,
        "send": cmd_send,
        "screenshot": cmd_screenshot,
        "health": cmd_health,
        "logs": cmd_logs,
    }

    return commands[args.command](args, bus_dir)


if __name__ == "__main__":
    sys.exit(main())
