#!/usr/bin/env python3
"""
Pluribus CUA TUI (Headless Dashboard) v0.4
==========================================
Computer-Use Agent dashboard for tmux/headless environments.
Displays real-time system health, bus metrics, STRp tasks, and provider status.

Features:
- SUPERMOTD ring status (Ring0-3)
- Bus event stream metrics
- STRp task pipeline status
- Provider health (Claude/Gemini/Codex/ChatGPT)
- Browser daemon VNC status
- Art Director mood metrics
- tmux session/pane integration
- Recent errors with context

Usage:
    python3 pluribus_cua_tui.py
    python3 pluribus_cua_tui.py --compact  # Single-line MOTD mode
    python3 pluribus_cua_tui.py --tmux     # Show tmux info only

Keybindings:
    q/Ctrl+C  - Quit
    r         - Force refresh
    v         - Toggle VNC info panel
    t         - Toggle tmux pane panel
"""
import sys
import time
import json
import subprocess
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Rich imports
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.console import Console, Group
from rich.text import Text
from rich.style import Style
from rich import box
from rich.columns import Columns
from rich.progress_bar import ProgressBar
from rich.align import Align

# Optional psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Configuration
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")).expanduser()
EVENTS_PATH = BUS_DIR / "events.ndjson"
BROWSER_STATE = Path("/pluribus/.pluribus/browser_daemon.json")
VPS_SESSION = Path("/pluribus/.pluribus/vps_session.json")
SUPERMOTD_SCRIPT = Path("/pluribus/nucleus/tools/supermotd.py")
PLURIBUS_ROOT = Path("/pluribus")

# Styles
STYLE_OK = Style(color="green", bold=True)
STYLE_WARN = Style(color="yellow")
STYLE_ERR = Style(color="red", bold=True)
STYLE_DIM = Style(dim=True)
STYLE_ACCENT = Style(color="cyan")
STYLE_RING0 = Style(color="magenta", bold=True)
STYLE_RING1 = Style(color="blue")
STYLE_RING2 = Style(color="cyan")
STYLE_RING3 = Style(color="green")


def get_system_metrics() -> dict:
    """Get CPU, memory, and load metrics."""
    if not HAS_PSUTIL:
        return {"cpu": 0, "mem": 0, "load": "N/A"}

    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory().percent
    load = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
    return {"cpu": cpu, "mem": mem, "load": f"{load:.2f}"}


def get_supermotd() -> dict:
    """Get ring status directly (faster than calling supermotd.py)."""
    # Build lightweight status without iterating entire bus
    rings = {
        "ring0": {"status": "sealed", "pqc_algorithm": "ML-DSA-65"},
        "ring1": {"lineage_id": "genesis", "generation": 0, "transfer_type": "VGT"},
        "ring2": {"infercells_active": 0},
        "ring3": {"omega_healthy": True, "omega_cycle": 0, "providers_available": [], "providers_total": 0},
    }

    # Check lineage
    lineage_path = PLURIBUS_ROOT / ".pluribus" / "lineage.json"
    if lineage_path.exists():
        try:
            data = json.loads(lineage_path.read_text())
            rings["ring1"]["lineage_id"] = data.get("lineage_id", "genesis")
            rings["ring1"]["generation"] = data.get("generation", 0)
        except:
            pass

    # Check infercells
    cells_dir = PLURIBUS_ROOT / ".pluribus" / "cells"
    if cells_dir.exists():
        try:
            rings["ring2"]["infercells_active"] = sum(1 for d in cells_dir.iterdir() if d.is_dir())
        except:
            pass

    # Check providers from vps_session
    if VPS_SESSION.exists():
        try:
            data = json.loads(VPS_SESSION.read_text())
            providers = data.get("providers", {})
            rings["ring3"]["providers_total"] = len(providers)
            rings["ring3"]["providers_available"] = [
                k for k, v in providers.items() if v.get("available", False)
            ]
            # Simple omega health check
            rings["ring3"]["omega_healthy"] = len(rings["ring3"]["providers_available"]) > 0
        except:
            pass

    return {"rings": rings}


def get_bus_metrics() -> dict:
    """Get bus event count and recent activity."""
    if not EVENTS_PATH.exists():
        return {"total": 0, "recent": 0, "last_topic": "N/A", "actors": set()}

    try:
        # Count total lines (fast via wc -l)
        result = subprocess.run(
            ["wc", "-l", str(EVENTS_PATH)],
            capture_output=True, text=True, timeout=2
        )
        total = int(result.stdout.split()[0]) if result.returncode == 0 else 0

        # Get last 50 events efficiently via tail
        result = subprocess.run(
            ["tail", "-n", "50", str(EVENTS_PATH)],
            capture_output=True, text=True, timeout=2
        )
        lines = result.stdout.strip().split('\n') if result.returncode == 0 else []

        recent_actors = set()
        last_topic = "N/A"
        recent_count = 0
        now = datetime.now(timezone.utc)

        for line in reversed(lines):
            if not line:
                continue
            try:
                ev = json.loads(line)
                recent_actors.add(ev.get("actor", "unknown"))
                if last_topic == "N/A":
                    last_topic = ev.get("topic", "N/A")

                # Count events in last 5 minutes
                ts = ev.get("ts", 0)
                if now.timestamp() - ts < 300:
                    recent_count += 1
            except:
                pass

        return {
            "total": total,
            "recent": recent_count,
            "last_topic": last_topic[:30],
            "actors": recent_actors
        }
    except Exception as e:
        return {"total": 0, "recent": 0, "last_topic": str(e), "actors": set()}


def get_provider_status() -> dict:
    """Get provider health from vps_session.json."""
    if not VPS_SESSION.exists():
        return {}

    try:
        data = json.loads(VPS_SESSION.read_text())
        providers = {}
        for pid, pdata in data.get("providers", {}).items():
            providers[pid] = {
                "available": pdata.get("available", False),
                "error": pdata.get("error"),
                "last_check": pdata.get("last_check_iso", "")[:19]
            }
        return providers
    except:
        return {}


def get_browser_status() -> dict:
    """Get browser daemon status."""
    if not BROWSER_STATE.exists():
        return {"running": False}

    try:
        data = json.loads(BROWSER_STATE.read_text())
        return {
            "running": data.get("running", False),
            "vnc_mode": data.get("vnc_mode", {}).get("enabled", False),
            "tabs": {
                k: v.get("status", "unknown")
                for k, v in data.get("tabs", {}).items()
            }
        }
    except:
        return {"running": False}


def get_strp_tasks(limit: int = 5) -> list:
    """Get recent STRp task status from bus."""
    tasks = []
    if not EVENTS_PATH.exists():
        return tasks

    # State mapping for STRp topics
    STATE_MAP = {
        "strp.worker.start": "starting",
        "strp.subagent.spawn": "spawning",
        "strp.output.grounding": "grounding",
        "strp.worker.item": "complete",
        "strp.worker.blocked": "blocked",
        "strp.worker.error": "error",
        "dialogos.submit": "submitted",
        "dialogos.output": "responded",
        "infercell.spawn": "spawning",
        "infercell.done": "done",
    }

    try:
        # Use tail for efficiency
        result = subprocess.run(
            ["tail", "-n", "300", str(EVENTS_PATH)],
            capture_output=True, text=True, timeout=2
        )
        lines = result.stdout.strip().split('\n') if result.returncode == 0 else []
        seen_reqs = set()

        for line in reversed(lines):
            if not line:
                continue
            try:
                ev = json.loads(line)
                topic = ev.get("topic", "")

                # Look for STRp-related events
                if any(x in topic for x in ["strp.", "task.", "infercell.", "dialogos."]):
                    data = ev.get("data", {})
                    req_id = data.get("req_id") or ev.get("run_id") or ev.get("id", "")[:8]
                    if req_id and req_id not in seen_reqs:
                        seen_reqs.add(req_id)

                        # Derive state from topic
                        state = STATE_MAP.get(topic, ev.get("kind", "event"))

                        # Get provider if present
                        provider = data.get("provider", data.get("model", ""))[:8]

                        tasks.append({
                            "id": req_id[:12],
                            "topic": topic[:25],
                            "status": state,
                            "provider": provider,
                            "ts": ev.get("iso", "")[-8:-1]  # HH:MM:SS
                        })
                        if len(tasks) >= limit:
                            break
            except:
                pass
    except:
        pass

    return tasks


def get_art_mood() -> dict:
    """Get mood metrics from Art Director (bus or state file)."""
    # Try to get latest art.state.update from bus
    if not EVENTS_PATH.exists():
        return {"mood": "neutral", "entropy": 0, "velocity": 0, "anxiety": 0}

    try:
        # Use tail + grep for efficiency
        result = subprocess.run(
            ["sh", "-c", f"tail -n 100 '{EVENTS_PATH}' | grep 'art.state.update' | tail -1"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            ev = json.loads(result.stdout.strip())
            data = ev.get("data", {})
            return {
                "mood": data.get("mood", "neutral"),
                "entropy": data.get("entropy", 0),
                "velocity": data.get("velocity", 0),
                "anxiety": data.get("anxiety", 0),
            }
    except:
        pass

    return {"mood": "neutral", "entropy": 0, "velocity": 0, "anxiety": 0}


def get_tmux_info() -> dict:
    """Get current tmux session, windows, and panes info."""
    result = {"in_tmux": False, "session": None, "windows": [], "panes": [], "active_pane": None}

    # Check if we're in tmux
    if not os.environ.get("TMUX"):
        return result

    result["in_tmux"] = True

    try:
        # Get current session
        sess_result = subprocess.run(
            ["tmux", "display-message", "-p", "#{session_name}"],
            capture_output=True, text=True, timeout=2
        )
        if sess_result.returncode == 0:
            result["session"] = sess_result.stdout.strip()

        # Get windows: #{window_index}:#{window_name}:#{window_active}
        win_result = subprocess.run(
            ["tmux", "list-windows", "-F", "#{window_index}:#{window_name}:#{window_active}"],
            capture_output=True, text=True, timeout=2
        )
        if win_result.returncode == 0:
            for line in win_result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split(':')
                if len(parts) >= 3:
                    result["windows"].append({
                        "index": parts[0],
                        "name": parts[1],
                        "active": parts[2] == "1"
                    })

        # Get panes: #{pane_index}:#{pane_pid}:#{pane_current_command}:#{pane_active}:#{pane_width}x#{pane_height}
        pane_result = subprocess.run(
            ["tmux", "list-panes", "-F", "#{pane_index}:#{pane_pid}:#{pane_current_command}:#{pane_active}:#{pane_width}x#{pane_height}"],
            capture_output=True, text=True, timeout=2
        )
        if pane_result.returncode == 0:
            for line in pane_result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split(':')
                if len(parts) >= 5:
                    pane_info = {
                        "index": parts[0],
                        "pid": parts[1],
                        "cmd": parts[2][:15],
                        "active": parts[3] == "1",
                        "size": parts[4]
                    }
                    result["panes"].append(pane_info)
                    if pane_info["active"]:
                        result["active_pane"] = pane_info

    except Exception:
        pass

    return result


def get_recent_errors(limit: int = 4) -> list:
    """Get recent error events from bus."""
    errors = []
    if not EVENTS_PATH.exists():
        return errors

    try:
        # Use tail + grep for efficiency
        result = subprocess.run(
            ["sh", "-c", f"tail -n 500 '{EVENTS_PATH}' | grep '\"level\":\"error\"' | tail -n {limit}"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in reversed(result.stdout.strip().split('\n')):
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    ts = ev.get("iso", "")[11:19]
                    msg = str(
                        ev.get("data", {}).get("error") or
                        ev.get("data", {}).get("message") or
                        ev.get("topic", "Unknown error")
                    )
                    errors.append(f"[dim]{ts}[/dim] {msg[:50]}")
                except:
                    pass
                if len(errors) >= limit:
                    break
    except:
        pass

    return errors


def build_header_panel(sys_metrics: dict, bus_metrics: dict, mood: dict) -> Panel:
    """Build the top header with ASCII art and system/bus metrics."""
    cpu_style = STYLE_OK if sys_metrics["cpu"] < 70 else STYLE_WARN if sys_metrics["cpu"] < 90 else STYLE_ERR
    mem_style = STYLE_OK if sys_metrics["mem"] < 70 else STYLE_WARN if sys_metrics["mem"] < 90 else STYLE_ERR

    # ASCII Art Header
    ascii_art = """
 [bold cyan]____  _  _  ____  ____  ____  __  __  ____[/] 
 [bold cyan](  _ \( \/ )(  _ \(  __)(  _ \(  )(  )/ ___)[/]
 [bold blue] ) __/ )  (  )   / ) _)  )   / )(__)( \___ \[/]
 [bold blue](__)  (_/\_)(__\_)(____)(__\_)(______)(____/[/]
    """
    
    header_content = Text()
    # header_content.append(ascii_art) # Add ASCII art
    # Center the ASCII art roughly
    
    metrics_line = Text()
    metrics_line.append("CPU:", style=STYLE_DIM)
    metrics_line.append(f"{sys_metrics['cpu']:4.0f}%", style=cpu_style)
    metrics_line.append(" │ ", style=STYLE_DIM)
    metrics_line.append("MEM:", style=STYLE_DIM)
    metrics_line.append(f"{sys_metrics['mem']:4.0f}%", style=mem_style)
    metrics_line.append(" │ ", style=STYLE_DIM)
    metrics_line.append("LOAD:", style=STYLE_DIM)
    metrics_line.append(f"{sys_metrics['load']}", style=STYLE_ACCENT)
    metrics_line.append(" │ ", style=STYLE_DIM)
    metrics_line.append("BUS:", style=STYLE_DIM)
    metrics_line.append(f"{bus_metrics['total']:,}", style=STYLE_ACCENT)
    
    mood_name = mood.get("mood", "neutral")
    metrics_line.append(" │ ", style=STYLE_DIM)
    metrics_line.append(f"MOOD: {mood_name}", style=STYLE_ACCENT)

    # Gestalt info (generation, lineage)
    lineage_path = PLURIBUS_ROOT / ".pluribus" / "lineage.json"
    gen = "—"
    lin_id = "—"
    try:
        if lineage_path.exists():
            lineage_data = json.loads(lineage_path.read_text(encoding="utf-8"))
            gen = lineage_data.get("generation", "—")
            lin_id = lineage_data.get("lineage_id", "")[:8] or "—"
    except Exception:
        pass

    metrics_line.append(" │ ", style=STYLE_DIM)
    metrics_line.append("GEN:", style=STYLE_DIM)
    metrics_line.append(f"{gen}", style=Style(color="cyan", bold=True))
    metrics_line.append(" │ ", style=STYLE_DIM)
    metrics_line.append("LIN:", style=STYLE_DIM)
    metrics_line.append(f"{lin_id}", style=STYLE_DIM)

    # Combine ASCII and metrics
    content = Group(
        Align.center(Text.from_markup(ascii_art)),
        Align.center(metrics_line)
    )

    return Panel(content, border_style="blue", padding=(0, 1))


def build_rings_panel(motd: dict) -> Panel:
    """Build ring status panel from SUPERMOTD."""
    rings = motd.get("rings", {})

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("Ring", style="bold")
    table.add_column("Status", style="dim")

    # Ring 0 - Security
    r0 = rings.get("ring0", {})
    r0_status = "sealed" if r0.get("status") == "sealed" else "UNSEALED!"
    r0_style = STYLE_RING0 if r0_status == "sealed" else STYLE_ERR
    table.add_row(Text("R0", style=r0_style), f"{r0_status} ({r0.get('pqc_algorithm', 'N/A')})")

    # Ring 1 - Lineage
    r1 = rings.get("ring1", {})
    table.add_row(Text("R1", style=STYLE_RING1), f"gen:{r1.get('generation', 0)} {r1.get('transfer_type', 'N/A')}")

    # Ring 2 - InferCells
    r2 = rings.get("ring2", {})
    table.add_row(Text("R2", style=STYLE_RING2), f"cells:{r2.get('infercells_active', 0)}")

    # Ring 3 - Omega
    r3 = rings.get("ring3", {})
    omega_ok = r3.get("omega_healthy", False)
    omega_style = STYLE_OK if omega_ok else STYLE_WARN
    table.add_row(Text("R3", style=STYLE_RING3), Text(f"ω:{'OK' if omega_ok else 'WARN'} cycle:{r3.get('omega_cycle', 0)}", style=omega_style))

    return Panel(table, title="[bold cyan]RINGS[/bold cyan]", border_style="cyan", height=7)


def build_providers_panel(providers: dict, browser: dict) -> Panel:
    """Build provider health panel."""
    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("Provider")
    table.add_column("Status")

    # Map nice names
    provider_names = {
        "claude-api": ("Claude API", "🟣"),
        "gemini-api": ("Gemini API", "✨"),
        "openai-api": ("OpenAI API", "💚"),
        "claude-cli": ("Claude CLI", "🟣"),
        "gemini-cli": ("Gemini CLI", "✨"),
        "chatgpt-web": ("ChatGPT Web", "💬"),
        "claude-web": ("Claude Web", "🟣"),
        "gemini-web": ("Gemini Web", "✨"),
    }

    for pid, pdata in providers.items():
        name, icon = provider_names.get(pid, (pid, "•"))
        available = pdata.get("available", False)
        status_text = "OK" if available else pdata.get("error", "ERR")[:15]
        style = STYLE_OK if available else STYLE_ERR
        table.add_row(f"{icon} {name}", Text(status_text, style=style))

    # Browser tabs
    if browser.get("running"):
        for tab, status in browser.get("tabs", {}).items():
            _, icon = provider_names.get(tab, (tab, "🌐"))
            style = STYLE_OK if status == "ready" else STYLE_WARN if status in ["needs_login", "needs_onboarding"] else STYLE_ERR
            table.add_row(f"{icon} {tab[:10]}", Text(status[:12], style=style))

    vnc_status = "[cyan]VNC[/cyan]" if browser.get("vnc_mode") else "[dim]headless[/dim]"

    return Panel(table, title=f"[bold green]PROVIDERS[/bold green] {vnc_status}", border_style="green", height=10)


def build_tasks_panel(tasks: list) -> Panel:
    """Build STRp tasks panel."""
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Topic", style="dim")
    table.add_column("State", width=10)
    table.add_column("Provider", style="dim", width=10)
    table.add_column("Time", style="dim", width=8)

    # State styling
    STATE_STYLES = {
        "complete": STYLE_OK,
        "done": STYLE_OK,
        "responded": STYLE_OK,
        "starting": STYLE_ACCENT,
        "spawning": STYLE_ACCENT,
        "submitted": STYLE_ACCENT,
        "grounding": STYLE_WARN,
        "blocked": STYLE_ERR,
        "error": STYLE_ERR,
    }

    if not tasks:
        table.add_row("[dim]No recent tasks[/dim]", "", "", "", "")
    else:
        for task in tasks:
            state = task.get("status", "")
            state_style = STATE_STYLES.get(state, STYLE_DIM)
            provider = task.get("provider", "")
            table.add_row(
                task["id"],
                task["topic"],
                Text(state, style=state_style),
                provider,
                task["ts"]
            )

    return Panel(table, title="[bold magenta]STRp PIPELINE[/bold magenta]", border_style="magenta")


def build_errors_panel(errors: list) -> Panel:
    """Build recent errors panel."""
    if not errors:
        content = Text("No recent errors", style=STYLE_DIM)
    else:
        content = Text("\n".join(errors))

    return Panel(content, title="[bold red]ERRORS[/bold red]", border_style="red", height=6)


def build_actors_panel(actors: set) -> Panel:
    """Build active actors panel."""
    if not actors:
        content = Text("No recent actors", style=STYLE_DIM)
    else:
        # Color-code known agents
        agent_styles = {
            "claude-opus": STYLE_RING0,
            "codex": STYLE_RING1,
            "gemini": STYLE_RING2,
            "root": STYLE_DIM,
        }
        content = Text()
        for i, actor in enumerate(sorted(actors)):
            if i > 0:
                content.append(" │ ", style=STYLE_DIM)
            style = agent_styles.get(actor, STYLE_ACCENT)
            content.append(actor[:12], style=style)

    return Panel(content, title="[bold yellow]ACTIVE AGENTS[/bold yellow]", border_style="yellow", height=3)


def build_tmux_panel(tmux_info: dict) -> Panel:
    """Build tmux session/panes panel."""
    if not tmux_info.get("in_tmux"):
        return Panel(
            Text("Not in tmux", style=STYLE_DIM),
            title="[bold blue]TMUX[/bold blue]",
            border_style="blue",
            height=8
        )

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("Info", style="dim")
    table.add_column("Value")

    # Session name
    session = tmux_info.get("session", "?")
    table.add_row("Session:", Text(session, style=STYLE_ACCENT))

    # Windows count
    windows = tmux_info.get("windows", [])
    active_win = next((w for w in windows if w.get("active")), None)
    win_display = f"{len(windows)} wins"
    if active_win:
        win_display = f"{active_win['index']}:{active_win['name']} ({len(windows)} total)"
    table.add_row("Window:", Text(win_display[:25], style=STYLE_OK))

    # Panes
    panes = tmux_info.get("panes", [])
    for pane in panes[:4]:  # Show max 4 panes
        idx = pane.get("index", "?")
        cmd = pane.get("cmd", "?")
        size = pane.get("size", "?")
        is_active = pane.get("active", False)

        pane_style = STYLE_ACCENT if is_active else STYLE_DIM
        marker = "►" if is_active else " "
        table.add_row(
            f"{marker}Pane {idx}:",
            Text(f"{cmd} ({size})", style=pane_style)
        )

    return Panel(table, title="[bold blue]TMUX[/bold blue]", border_style="blue", height=8)


def make_layout() -> Layout:
    """Create the dashboard layout."""
    layout = Layout()

    # Top row: header
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )

    # Main area: left (rings + providers) | center (tasks + tmux) | right (errors)
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="center", ratio=2),
        Layout(name="right", ratio=1)
    )

    # Left column: rings on top, providers below
    layout["left"].split_column(
        Layout(name="rings", size=7),
        Layout(name="providers")
    )

    # Center column: tasks on top, tmux below
    layout["center"].split_column(
        Layout(name="tasks"),
        Layout(name="tmux", size=8)
    )

    return layout


def update_dashboard(layout: Layout):
    """Update all dashboard panels."""
    # Gather data
    sys_metrics = get_system_metrics()
    bus_metrics = get_bus_metrics()
    motd = get_supermotd()
    providers = get_provider_status()
    browser = get_browser_status()
    tasks = get_strp_tasks()
    errors = get_recent_errors()
    mood = get_art_mood()
    tmux_info = get_tmux_info()

    # Update panels
    layout["header"].update(build_header_panel(sys_metrics, bus_metrics, mood))
    layout["rings"].update(build_rings_panel(motd))
    layout["providers"].update(build_providers_panel(providers, browser))
    layout["tasks"].update(build_tasks_panel(tasks))
    layout["tmux"].update(build_tmux_panel(tmux_info))
    layout["right"].update(build_errors_panel(errors))
    layout["footer"].update(build_actors_panel(bus_metrics.get("actors", set())))


def run_compact_mode():
    """Single-line MOTD output for prompt integration."""
    sys_m = get_system_metrics()
    bus_m = get_bus_metrics()
    motd = get_supermotd()
    mood = get_art_mood()

    rings = motd.get("rings", {})
    r3 = rings.get("ring3", {})
    omega = "ω✓" if r3.get("omega_healthy") else "ω!"

    mood_icons = {"neutral": "◉", "calm": "◎", "active": "◈", "busy": "◆", "anxious": "◇", "critical": "◆!", "hyper": "⚡"}
    mood_icon = mood_icons.get(mood.get("mood", "neutral"), "◉")

    print(f"CPU:{sys_m['cpu']:.0f}% MEM:{sys_m['mem']:.0f}% BUS:{bus_m['total']:,} {omega} {mood_icon} │ {bus_m['last_topic']}")


def run_tmux_mode():
    """Display tmux info only."""
    tmux_info = get_tmux_info()
    console = Console()

    if not tmux_info.get("in_tmux"):
        console.print("[dim]Not running in tmux[/dim]")
        return

    console.print(f"[bold cyan]Session:[/bold cyan] {tmux_info.get('session', '?')}")

    # Windows
    windows = tmux_info.get("windows", [])
    win_str = " ".join(
        f"[{'green' if w.get('active') else 'dim'}]{w['index']}:{w['name']}[/]"
        for w in windows
    )
    console.print(f"[bold cyan]Windows:[/bold cyan] {win_str}")

    # Panes
    panes = tmux_info.get("panes", [])
    for pane in panes:
        marker = "[green]►[/green]" if pane.get("active") else " "
        console.print(f" {marker} Pane {pane['index']}: {pane['cmd']} ({pane['size']})")


def main():
    """Main entry point."""
    # Check for compact mode
    if "--compact" in sys.argv:
        run_compact_mode()
        return 0

    # Check for tmux mode
    if "--tmux" in sys.argv:
        run_tmux_mode()
        return 0

    console = Console()
    layout = make_layout()

    console.print("[bold cyan]Pluribus CUA TUI v0.3[/bold cyan] - Press Ctrl+C to exit\n")

    try:
        with Live(layout, refresh_per_second=2, screen=True, console=console):
            while True:
                update_dashboard(layout)
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
