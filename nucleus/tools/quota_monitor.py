#!/usr/bin/env python3
"""
quota_monitor.py - Real-Time Quota Monitoring Daemon

Version: 1.0.0
Ring: 1 (Services)
Protocol: Quota Protocol v1 / DKIN v30

This daemon runs in the PBTSO monitor pane and provides:
  1. Real-time quota tracking across all providers
  2. Automatic provider CLI status polling
  3. Threshold alerts and bus event emission
  4. WIP meter visualization

Integration with PBTSO:
  - Spawned in the 'monitor' pane by tmux_swarm_orchestrator.py
  - Polls /status from active provider CLIs
  - Emits quota.* bus events for observability

Usage:
    python3 quota_monitor.py [--interval 30] [--providers claude,codex,gemini]

Semops: PBQMONITOR

@module tools/quota_monitor
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import QuotaManager
sys.path.insert(0, str(Path(__file__).parent))
from quota_manager import QuotaManager, Provider, MeteringShape, GatingDecision

VERSION = "1.1.0"

# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def get_cli_status(provider: Provider) -> Optional[str]:
    """Query the CLI status for a provider."""
    cli_map = {
        Provider.CLAUDE: ["claude", "/status"],
        Provider.CODEX: ["codex", "/status"],
        Provider.GEMINI: ["gemini", "quota"],
    }

    if provider not in cli_map:
        return None

    cmd = cli_map[provider]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def format_meter(pct: float, width: int = 20) -> str:
    """Format a WIP meter bar."""
    filled = int((pct / 100) * width)
    empty = width - filled

    if pct >= 50:
        color = Colors.GREEN
    elif pct >= 25:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    return f"{color}[{'#' * filled}{'-' * empty}]{Colors.RESET} {pct:.1f}%"


def format_provider_status(qm: QuotaManager, provider: Provider) -> str:
    """Format status for a single provider."""
    state = qm.get_budget_state(provider)
    shape = qm.get_metering_shape(provider)

    lines = []
    name = provider.value.upper()

    # Header with color based on health
    if state.remaining_session_pct > 50:
        color = Colors.GREEN
    elif state.remaining_session_pct > 15:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    lines.append(f"{color}{Colors.BOLD}[{name}]{Colors.RESET}")

    if shape == MeteringShape.REQUEST_COUNTED:
        lines.append(f"  RPM:   {format_meter(min(100, state.remaining_burst / 60 * 100))}")
        lines.append(f"  Daily: {format_meter(state.remaining_long_pct)}")
        lines.append(f"         {state.remaining_long:.0f} requests remaining")
    else:
        lines.append(f"  5h:    {format_meter(state.remaining_session_pct)}")
        lines.append(f"         {state.remaining_session:.1f} units remaining")
        lines.append(f"  Week:  {format_meter(state.remaining_long_pct)}")

    # Gating decision
    result = qm.can_proceed(provider, tokens_in=5000)
    decision_color = {
        GatingDecision.PROCEED: Colors.GREEN,
        GatingDecision.DOWNGRADE: Colors.YELLOW,
        GatingDecision.QUEUE: Colors.YELLOW,
        GatingDecision.REJECT: Colors.RED,
    }.get(result.decision, Colors.RESET)

    lines.append(f"  Gate:  {decision_color}{result.decision.value}{Colors.RESET}")

    return "\n".join(lines)


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name == 'posix' else 'cls')


def run_monitor(
    interval: int = 30,
    providers: Optional[list[Provider]] = None,
    tier: str = "pro"
):
    """Run the quota monitor daemon."""
    qm = QuotaManager(tier=tier)
    providers = providers or [Provider.CLAUDE, Provider.CODEX, Provider.GEMINI]

    print(f"{Colors.BOLD}QUOTA MONITOR v{VERSION}{Colors.RESET}")
    print(f"Monitoring: {', '.join(p.value for p in providers)}")
    print(f"Interval: {interval}s | Tier: {tier}")
    print("-" * 60)

    iteration = 0

    while True:
        try:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Clear and redraw
            clear_screen()

            print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
            print(f"{Colors.CYAN}PLURIBUS QUOTA MONITOR{Colors.RESET} | {timestamp} | iter:{iteration}")
            print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
            print()

            # Poll CLI status for each provider
            for provider in providers:
                # Try to get fresh status from CLI
                status_output = get_cli_status(provider)
                if status_output:
                    qm.update_from_status(provider, status_output)

                # Display status
                print(format_provider_status(qm, provider))
                print()

            # Optimal provider recommendation
            print(f"{Colors.BOLD}--- RECOMMENDATION ---{Colors.RESET}")
            optimal, result = qm.select_optimal_provider()
            print(f"Optimal Provider: {Colors.GREEN}{optimal.value}{Colors.RESET}")
            print(f"Reason: {result.reason}")

            if result.recommended_model:
                print(f"Suggested Model: {Colors.YELLOW}{result.recommended_model}{Colors.RESET}")

            print()

            # Optimization tips
            print(f"{Colors.DIM}--- OPTIMIZATION TIPS ---{Colors.RESET}")
            for rule in qm.get_optimization_rules(optimal)[:3]:
                print(f"{Colors.DIM}  - {rule}{Colors.RESET}")

            print()
            print(f"{Colors.DIM}Next update in {interval}s... (Ctrl+C to exit){Colors.RESET}")

            time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Monitor stopped.{Colors.RESET}")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            time.sleep(5)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time quota monitoring daemon"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Update interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--providers", "-p",
        type=str,
        default="claude,codex,gemini",
        help="Comma-separated list of providers to monitor"
    )
    parser.add_argument(
        "--tier", "-t",
        type=str,
        default="pro",
        help="Quota tier (default: pro)"
    )
    parser.add_argument(
        "--once", "-1",
        action="store_true",
        help="Run once and exit (for scripting)"
    )

    args = parser.parse_args()

    providers = [Provider(p.strip()) for p in args.providers.split(",")]

    if args.once:
        qm = QuotaManager(tier=args.tier)
        print(qm.format_status_report())
    else:
        run_monitor(
            interval=args.interval,
            providers=providers,
            tier=args.tier
        )


if __name__ == "__main__":
    main()
