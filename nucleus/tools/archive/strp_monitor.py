#!/usr/bin/env python3
"""STRp Control Center TUI - 'Stir the Pot'.

Interactive monitoring and control dashboard for the Pluribus STRp pipeline.
Uses Textual library for terminal UI.

Usage:
    python3 nucleus/tools/strp_monitor.py [--root /path/to/.pluribus]

Panels:
    1. Event Log - tails .pluribus/bus/events.ndjson
    2. Pending Requests - shows .pluribus/index/requests.ndjson
    3. Agent Status - displays pluribus.check.report events
    4. Command Input - send custom events to the bus

Keybindings:
    q - Quit
    r - Refresh
    f - Toggle topic filter
    c - Trigger curation loop
    w - Spawn worker (strp_worker.py)
    x - Clear selected agent status (set to idle)
    v - View Services TUI
    p - View VPS Session status
    m - Set flow mode to Monitor/Approve
    a - Set flow mode to Automatic
    d - Toggle Dashboard supervisor (VPS daemon + Bridge)
    / - Focus command input
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical
    from textual.reactive import reactive
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        Log,
        Static,
    )
    from textual.timer import Timer
    from textual.binding import Binding
except ImportError:
    sys.stderr.write("ERROR: textual not installed. Run: pip install textual\n")
    sys.exit(1)


# Worker heartbeat timeout - consider worker inactive if no events in this many seconds
WORKER_HEARTBEAT_TIMEOUT = 30.0


@dataclass
class AgentStatus:
    """Parsed agent status from pluribus.check.report event."""
    actor: str
    status: str = "unknown"
    health: str = "unknown"
    queue_depth: int = 0
    current_task: str = ""
    blockers: list[str] = field(default_factory=list)
    vor_cdi: float | None = None
    vor_passed: int | None = None
    vor_failed: int | None = None
    last_seen_iso: str = ""


@dataclass
class WorkerInfo:
    """Tracking info for strp_worker processes."""
    actor: str
    last_seen: float  # time.time() timestamp
    last_topic: str = ""
    pid: int | None = None
    provider: str = ""
    is_spawned: bool = False  # True if we spawned this worker


def find_pluribus_root(start: Path) -> Path | None:
    """Find .pluribus directory by walking upward."""
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand / ".pluribus"
        if (cand / ".pluribus").is_dir():
            return cand / ".pluribus"
    return None


def parse_ndjson_file(path: Path, limit: int = 100) -> list[dict]:
    """Read last N lines from NDJSON file."""
    if not path.exists():
        return []
    lines = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        return []
    return lines[-limit:]


def tail_ndjson(path: Path, last_pos: int = 0) -> tuple[list[dict], int]:
    """Read new lines from NDJSON file since last_pos."""
    if not path.exists():
        return [], 0
    try:
        size = path.stat().st_size
        if size < last_pos:
            last_pos = 0  # file truncated, restart
        if size == last_pos:
            return [], last_pos
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(last_pos)
            new_lines = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        new_lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return new_lines, f.tell()
    except Exception:
        return [], last_pos


def level_style(level: str) -> str:
    """Return rich style for log level."""
    return {
        "debug": "dim",
        "info": "green",
        "warn": "yellow",
        "error": "bold red",
    }.get(level, "")


VALID_BUS_KINDS: set[str] = {"log", "request", "response", "artifact", "metric"}
BUS_KIND_ALIASES: dict[str, str] = {
    # Older/looser terminology used in docs and UI affordances.
    "command": "request",
    "control": "request",
    "event": "log",
}


def normalize_bus_kind(kind: str) -> str | None:
    k = (kind or "").strip().lower()
    if not k:
        return None
    k = BUS_KIND_ALIASES.get(k, k)
    if k not in VALID_BUS_KINDS:
        return None
    return k


def normalize_bus_level(level: str) -> str:
    lv = (level or "").strip().lower() or "info"
    if lv == "warning":
        lv = "warn"
    if lv not in {"debug", "info", "warn", "error"}:
        return "info"
    return lv


def format_event(event: dict) -> str:
    """Format event for log display with dimensional richness."""
    iso = event.get("iso", "")[:19]
    topic = event.get("topic", "?")
    actor = event.get("actor", "?")
    level = event.get("level", "info")
    kind = event.get("kind", "log")

    # Build parts list
    parts = []

    # Dimensional layers (Big Events Rewrite)
    semantic = event.get("semantic", {})
    geometric = event.get("geometric", {})
    omega = event.get("omega", {})
    auom = event.get("auom", {})
    topology = event.get("topology", {})
    evolutionary = event.get("evolutionary", {})

    # Impact badge
    impact = semantic.get("impact") if isinstance(semantic, dict) else None
    if impact:
        impact_style = {
            "critical": "bold magenta",
            "high": "bold yellow",
            "medium": "cyan",
            "low": "dim"
        }.get(impact, "dim")
        parts.append(f"[{impact_style}][{impact.upper()}][/{impact_style}]")

    # Omega class
    omega_class = omega.get("omega_class") if isinstance(omega, dict) else None
    if omega_class and omega_class != "prima":
        omega_style = "bold yellow" if omega_class == "omega" else "magenta"
        parts.append(f"[{omega_style}]Ω:{omega_class}[/{omega_style}]")

    # Topology (if not single)
    topo = topology.get("topology") if isinstance(topology, dict) else None
    fanout = topology.get("fanout", 1) if isinstance(topology, dict) else 1
    if topo and topo != "single":
        parts.append(f"[cyan]{topo}×{fanout}[/cyan]")

    # AUOM compliance
    if isinstance(auom, dict) and "compliant" in auom:
        auom_mark = "[green]✓AUOM[/green]" if auom["compliant"] else "[red]✗AUOM[/red]"
        parts.append(auom_mark)

    # Geometric mean
    gm = geometric.get("geometric_mean") if isinstance(geometric, dict) else None
    if gm is not None:
        gm_style = "green" if gm >= 0.8 else "yellow" if gm >= 0.5 else "red"
        parts.append(f"[{gm_style}]GM:{gm:.2f}[/{gm_style}]")

    # VGT/HGT transfer
    transfer = evolutionary.get("transfer_type") if isinstance(evolutionary, dict) else None
    if transfer and transfer != "none":
        transfer_style = "bold yellow" if transfer == "HGT" else "green"
        parts.append(f"[{transfer_style}]{transfer}[/{transfer_style}]")

    # Build dimensional suffix
    dim_suffix = " ".join(parts)
    if dim_suffix:
        dim_suffix = " " + dim_suffix

    # Semantic summary (if different from topic)
    summary = semantic.get("summary") if isinstance(semantic, dict) else None
    summary_line = ""
    if summary and summary != topic and len(summary) < 60:
        summary_line = f"\n    [italic]{summary}[/italic]"

    # Legacy data summary fallback
    data_summary = ""
    data = event.get("data")
    if isinstance(data, dict) and not summary:
        if "goal" in data and data["goal"]:
            data_summary = f" goal={str(data['goal'])[:40]}..."
        elif "status" in data and data["status"]:
            data_summary = f" status={data['status']}"
        elif "note" in data and data["note"]:
            data_summary = f" note={str(data['note'])[:40]}"

    style = level_style(level)
    main_line = f"[{style}]{iso} [{kind}] {topic} ({actor}){data_summary}{dim_suffix}[/{style}]"
    return main_line + summary_line


def format_request(req: dict) -> tuple:
    """Format request for table row."""
    req_id = req.get("req_id", "?")[:8]
    iso = req.get("iso", "")[:16]
    kind = req.get("kind", "?")
    actor = req.get("actor", "?")
    goal = req.get("goal", "")[:50]
    return (req_id, iso, kind, actor, goal)


def parse_agent_status(event: dict) -> AgentStatus | None:
    """Parse pluribus.check.report event into AgentStatus."""
    if event.get("topic") != "pluribus.check.report":
        return None
    data = event.get("data", {})
    if not isinstance(data, dict):
        return None
    actor = event.get("actor", "unknown")
    current = data.get("current_task", {})
    vor = data.get("vor_metrics", {})
    return AgentStatus(
        actor=actor,
        status=data.get("status", "unknown"),
        health=data.get("health", "unknown"),
        queue_depth=data.get("queue_depth", 0),
        current_task=current.get("goal", "") if isinstance(current, dict) else "",
        blockers=data.get("blockers", []),
        vor_cdi=vor.get("cdi") if vor else None,
        vor_passed=vor.get("passed") if vor else None,
        vor_failed=vor.get("failed") if vor else None,
        last_seen_iso=event.get("iso", "") or event.get("ts", ""),
    )


def is_worker_event(event: dict) -> bool:
    """Check if event is from strp_worker."""
    topic = event.get("topic", "")
    return topic.startswith("strp.worker.") or topic.startswith("agent.coord.") or topic.startswith("agent.topology.")


def parse_worker_event(event: dict) -> WorkerInfo | None:
    """Parse worker-related event into WorkerInfo."""
    if not is_worker_event(event):
        return None
    actor = event.get("actor", "unknown")
    data = event.get("data", {})
    provider = ""
    if isinstance(data, dict):
        provider = data.get("provider", "")
    return WorkerInfo(
        actor=actor,
        last_seen=time.time(),
        last_topic=event.get("topic", ""),
        provider=provider,
    )


class STRpControlCenter(App):
    """STRp Control Center TUI - 'Stir the Pot'."""

    # Force dark mode for better visibility
    dark = True

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 3;
        grid-columns: 2fr 1fr 1fr;
        grid-rows: 2fr 1fr auto;
    }

    #event-log-container {
        border: solid green;
        height: 100%;
    }

    #requests-panel {
        border: solid blue;
        height: 100%;
    }

    #stack-panel {
        border: solid cyan;
        height: 100%;
    }

    #agent-panel {
        column-span: 3;
        border: solid yellow;
        height: 100%;
    }

    #control-panel {
        column-span: 3;
        border: solid magenta;
        height: auto;
        padding: 0 1;
    }

    #control-row {
        height: 3;
        width: 100%;
    }

    #button-row {
        height: 3;
        width: 100%;
    }

    #command-input {
        width: 1fr;
    }

    #send-btn {
        width: 12;
        margin-left: 1;
    }

    #curation-btn {
        width: 18;
        margin-left: 1;
    }

    #worker-btn {
        width: 20;
        margin-left: 1;
    }

    #clear-agent-btn {
        width: 22;
        margin-left: 1;
    }

    #status-bar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }

    #worker-status {
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    .panel-title {
        text-style: bold;
        background: $surface;
        padding: 0 1;
    }

    .control-title {
        text-style: bold;
        color: magenta;
    }

    DataTable {
        height: 100%;
    }

    Log {
        height: 100%;
    }

    .hint {
        color: $text-muted;
        text-style: italic;
    }

    .worker-active {
        color: green;
    }

    .worker-inactive {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("f", "toggle_filter", "Filter"),
        Binding("c", "trigger_curation", "Curate"),
        Binding("w", "spawn_worker", "Worker"),
        Binding("x", "clear_agent_status", "Clear"),
        Binding("v", "view_services", "Services"),
        Binding("p", "view_vps", "VPS"),
        Binding("m", "set_mode_monitor", "Mode:m"),
        Binding("a", "set_mode_auto", "Mode:A"),
        Binding("d", "toggle_dashboard", "Dashboard"),
        Binding("slash", "focus_input", "Command", key_display="/"),
        Binding("escape", "unfocus_input", "Unfocus", show=False),
    ]

    TITLE = "STRp Control Center"
    SUB_TITLE = "Stir the Pot"

    def __init__(self, pluribus_root: Path):
        super().__init__()
        self.pluribus_root = pluribus_root
        self.events_path = pluribus_root / "bus" / "events.ndjson"
        self.requests_path = pluribus_root / "index" / "requests.ndjson"
        self.bus_tool = Path(__file__).parent / "agent_bus.py"
        self.curation_tool = Path(__file__).parent / "strp_curation_loop.py"
        self.worker_tool = Path(__file__).parent / "strp_worker.py"
        self.vps_tool = Path(__file__).parent / "vps_session.py"
        self.supervisor_tool = Path(__file__).parent / "dashboard_supervisor.py"
        self.supervisor_proc: subprocess.Popen | None = None
        self.events_pos = 0
        self.agent_statuses: dict[str, AgentStatus] = {}
        self.worker_infos: dict[str, WorkerInfo] = {}
        self.spawned_workers: list[subprocess.Popen] = []
        self.requests_mtime = 0.0
        self.topic_filter: str | None = None
        self.curation_running = False
        self.selected_agent: str | None = None
        # VPS Session state
        self.vps_flow_mode: str = "m"
        self.vps_providers: dict[str, dict] = {
            "gemini": {"available": False},
            "claude": {"available": False},
            "codex": {"available": False},
            "vertex": {"available": False},
            "mock": {"available": True},
        }
        self.vps_active_fallback: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Event Log Panel
        yield Vertical(
            Static("Event Log", classes="panel-title"),
            Log(id="event-log", highlight=True, max_lines=500),
            id="event-log-container",
        )

        # Pending Requests Panel
        yield Vertical(
            Static("Pending Requests", classes="panel-title"),
            DataTable(id="requests-table"),
            id="requests-panel",
        )

        # Pushdown Stack Panel
        yield Vertical(
            Static("d-subautomata Stack (Pushdown)", classes="panel-title"),
            DataTable(id="stack-table"),
            id="stack-panel",
        )

        # Agent Status Panel
        yield Vertical(
            Static("Agent Status (pluribus.check.report) - Press x to clear selected", classes="panel-title"),
            DataTable(id="agent-table"),
            id="agent-panel",
        )

        # Control Panel
        yield Vertical(
            Static("Command Center", classes="panel-title control-title"),
            Horizontal(
                Input(
                    placeholder="topic:kind:data (e.g. pluribus.check.trigger:request:{} or strp.request:task:{\"goal\":\"...\"}) ",
                    id="command-input",
                ),
                Button("Send", id="send-btn", variant="primary"),
                id="control-row",
            ),
            Horizontal(
                Button("Curate (c)", id="curation-btn", variant="success"),
                Button("Spawn Worker (w)", id="worker-btn", variant="warning"),
                Button("Clear Agent (x)", id="clear-agent-btn", variant="default"),
                Button("Services (v)", id="services-btn", variant="primary"),
                id="button-row",
            ),
            Static("", id="worker-status"),
            Static("", id="status-bar"),
            id="control-panel",
        )

        yield Footer()

    def on_mount(self) -> None:
        # Initialize requests table
        req_table = self.query_one("#requests-table", DataTable)
        req_table.add_columns("ID", "Time", "Kind", "Actor", "Goal")

        # Initialize stack table
        stack_table = self.query_one("#stack-table", DataTable)
        stack_table.add_columns("Depth", "Item", "State")

        # Initialize agent table with cursor for selection
        agent_table = self.query_one("#agent-table", DataTable)
        agent_table.cursor_type = "row"
        agent_table.add_columns(
            "Actor", "Status", "Health", "Queue", "Task", "VOR CDI", "Last Seen"
        )

        # Load initial data
        self.load_initial_events()
        self.refresh_requests()
        self.refresh_agents()

        # Set up polling timers
        self.set_interval(0.5, self.poll_events)
        self.set_interval(2.0, self.refresh_requests)
        self.set_interval(1.0, self.update_worker_status)

        # Update status bar
        self.update_status("Ready. Press / to enter command, c to curate, w to spawn worker.")

    def update_status(self, message: str) -> None:
        """Update the status bar message."""
        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(f"[dim]{message}[/dim]")

    def update_worker_status(self) -> None:
        """Update the worker status indicator."""
        worker_status = self.query_one("#worker-status", Static)

        # Check for active workers
        now = time.time()
        active_workers = []
        inactive_workers = []

        for actor, info in self.worker_infos.items():
            age = now - info.last_seen
            if age <= WORKER_HEARTBEAT_TIMEOUT:
                active_workers.append((actor, info, age))
            else:
                inactive_workers.append((actor, info, age))

        # Also check spawned processes
        spawned_alive = 0
        for proc in self.spawned_workers[:]:
            if proc.poll() is None:
                spawned_alive += 1
            else:
                self.spawned_workers.remove(proc)

        if active_workers:
            worker_list = ", ".join(
                f"{actor}({info.provider or 'auto'})"
                for actor, info, _ in active_workers[:3]
            )
            if len(active_workers) > 3:
                worker_list += f" +{len(active_workers)-3} more"
            worker_status.update(
                f"[green bold]Workers Active:[/green bold] [green]{worker_list}[/green]"
                + (f" | [yellow]Spawned: {spawned_alive}[/yellow]" if spawned_alive else "")
            )
        elif spawned_alive:
            worker_status.update(
                f"[yellow]Spawned Workers: {spawned_alive} (waiting for heartbeat...)[/yellow]"
            )
        else:
            worker_status.update(
                f"[dim]No active workers detected. Press 'w' to spawn one.[/dim]"
            )

    def load_initial_events(self) -> None:
        """Load last 100 events on startup."""
        events = parse_ndjson_file(self.events_path, limit=100)
        log_widget = self.query_one("#event-log", Log)
        for event in events:
            if self.topic_filter and not event.get("topic", "").startswith(self.topic_filter):
                continue
            log_widget.write_line(format_event(event))
            # Track agent status
            status = parse_agent_status(event)
            if status:
                self.agent_statuses[status.actor] = status
            # Track worker status
            worker_info = parse_worker_event(event)
            if worker_info:
                existing = self.worker_infos.get(worker_info.actor)
                if existing:
                    existing.last_seen = worker_info.last_seen
                    existing.last_topic = worker_info.last_topic
                    if worker_info.provider:
                        existing.provider = worker_info.provider
                else:
                    self.worker_infos[worker_info.actor] = worker_info
        # Set position to end of file
        if self.events_path.exists():
            self.events_pos = self.events_path.stat().st_size

    def poll_events(self) -> None:
        """Poll for new events."""
        new_events, new_pos = tail_ndjson(self.events_path, self.events_pos)
        if new_events:
            self.events_pos = new_pos
            log_widget = self.query_one("#event-log", Log)
            for event in new_events:
                if self.topic_filter and not event.get("topic", "").startswith(self.topic_filter):
                    continue
                log_widget.write_line(format_event(event))
                # Track agent status
                status = parse_agent_status(event)
                if status:
                    self.agent_statuses[status.actor] = status
                    self.refresh_agents()
                
                # Track stack updates
                if event.get("topic") == "omega.vector.stream":
                    self.update_stack(event)

                # Track worker status
                worker_info = parse_worker_event(event)
                if worker_info:
                    existing = self.worker_infos.get(worker_info.actor)
                    if existing:
                        existing.last_seen = worker_info.last_seen
                        existing.last_topic = worker_info.last_topic
                        if worker_info.provider:
                            existing.provider = worker_info.provider
                    else:
                        self.worker_infos[worker_info.actor] = worker_info

    def update_stack(self, event: dict) -> None:
        """Update the stack visualization from an omega event."""
        data = event.get("data", {})
        if not isinstance(data, dict):
            return
        
        # We expect data to have "stack_top", "stack_depth"
        # Ideally, we want the whole stack list, but if we only get top, we just show that.
        # Let's assume the event sends the full stack list for visualization if available,
        # or we just show the top.
        
        # Simulating a stack list if only top provided
        stack_items = data.get("stack", [])
        if not stack_items and "stack_top" in data:
            stack_items = [{"item": data["stack_top"], "depth": data.get("stack_depth", 0), "state": "active"}]
            
        table = self.query_one("#stack-table", DataTable)
        table.clear()
        
        # If we have a geometric mean, show it as a "header" row or just first row?
        gm = data.get("geometric_mean")
        if gm is not None:
             table.add_row("Ω", f"Geometric Mean: {gm:.4f}", "")

        for item in reversed(stack_items):
            if isinstance(item, dict):
                table.add_row(str(item.get("depth", "?")), item.get("item", "?"), item.get("state", "?"))
            else:
                table.add_row("-", str(item), "-")

    def refresh_requests(self) -> None:
        """Refresh pending requests table."""
        if not self.requests_path.exists():
            return
        try:
            mtime = self.requests_path.stat().st_mtime
            if mtime == self.requests_mtime:
                return
            self.requests_mtime = mtime
        except Exception:
            return

        requests = parse_ndjson_file(self.requests_path, limit=50)
        table = self.query_one("#requests-table", DataTable)
        table.clear()
        for req in reversed(requests):  # newest first
            table.add_row(*format_request(req))

    def refresh_agents(self) -> None:
        """Refresh agent status table."""
        table = self.query_one("#agent-table", DataTable)
        table.clear()
        for actor, status in sorted(self.agent_statuses.items()):
            vor_cdi = f"{status.vor_cdi:.2f}" if status.vor_cdi is not None else "-"
            table.add_row(
                actor,
                status.status,
                status.health,
                str(status.queue_depth),
                status.current_task[:30] if status.current_task else "-",
                vor_cdi,
                str(status.last_seen_iso)[:16],
                key=actor,
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in agent table."""
        if event.data_table.id == "agent-table":
            # Get the actor name from the row key
            if event.row_key:
                self.selected_agent = str(event.row_key.value)
                self.update_status(f"Selected agent: {self.selected_agent}. Press 'x' to clear status.")

    def emit_bus_event(self, topic: str, kind: str, data: Any, level: str = "info", actor: str = "strp_control_center") -> bool:
        """Emit an event to the bus using agent_bus.py."""
        if not self.bus_tool.exists():
            self.notify("agent_bus.py not found!", severity="error")
            return False

        normalized_kind = normalize_bus_kind(kind)
        if not normalized_kind:
            self.notify(
                f"Invalid kind '{kind}'. Use one of: {', '.join(sorted(VALID_BUS_KINDS))}",
                severity="error",
            )
            return False

        level = normalize_bus_level(level)

        data_str = json.dumps(data) if not isinstance(data, str) else data

        cmd = [
            sys.executable,
            str(self.bus_tool),
            "pub",
            "--topic", topic,
            "--kind", normalized_kind,
            "--level", level,
            "--data", data_str,
            "--actor", actor,
        ]

        # Set bus directory
        bus_dir = self.pluribus_root / "bus"
        env = os.environ.copy()
        env["PLURIBUS_BUS_DIR"] = str(bus_dir)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                env=env,
            )
            if result.returncode == 0:
                return True
            else:
                self.notify(f"Bus error: {result.stderr[:100]}", severity="error")
                return False
        except subprocess.TimeoutExpired:
            self.notify("Bus command timed out", severity="error")
            return False
        except Exception as e:
            self.notify(f"Bus error: {e}", severity="error")
            return False

    def parse_command(self, command: str) -> tuple[str, str, Any] | None:
        """Parse command string into (topic, kind, data).

        Formats supported:
        - topic:kind:data (e.g. pluribus.check.trigger:request:{})
        - topic:kind (data defaults to {})
        - topic (kind defaults to 'request', data to {})
        """
        command = command.strip()
        if not command:
            return None

        # Try to parse as topic:kind:data
        parts = command.split(":", 2)

        topic = parts[0].strip()
        kind_raw = parts[1] if len(parts) > 1 else "request"
        kind = normalize_bus_kind(kind_raw)
        if not topic or not kind:
            return None
        data_str = parts[2] if len(parts) > 2 else "{}"

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            # If not valid JSON, treat as string note
            data = {"note": data_str}

        return (topic, kind, data)

    def action_refresh(self) -> None:
        """Manual refresh."""
        self.refresh_requests()
        self.refresh_agents()
        self.notify("Refreshed")

    def action_toggle_filter(self) -> None:
        """Toggle topic filter (cycles through common prefixes)."""
        filters = [None, "strp.", "pluribus.check", "curation.", "kg.", "rag.", "agent."]
        current_idx = filters.index(self.topic_filter) if self.topic_filter in filters else 0
        self.topic_filter = filters[(current_idx + 1) % len(filters)]
        self.notify(f"Filter: {self.topic_filter or 'none'}")

    def action_focus_input(self) -> None:
        """Focus the command input."""
        self.query_one("#command-input", Input).focus()
        self.update_status("Enter command: topic:kind:data | Press Enter to send, Escape to cancel")

    def action_unfocus_input(self) -> None:
        """Unfocus the command input."""
        self.set_focus(None)
        self.update_status("Ready. Press / to enter command, c to curate, w to spawn worker.")

    def action_trigger_curation(self) -> None:
        """Trigger the curation loop in the background."""
        if self.curation_running:
            self.notify("Curation already running!", severity="warning")
            return

        if not self.curation_tool.exists():
            self.notify("strp_curation_loop.py not found!", severity="error")
            return

        self.curation_running = True
        self.update_status("Starting curation loop...")

        # Emit event to bus announcing curation trigger
        self.emit_bus_event(
            "strp.control.curation_triggered",
            "request",
            {"source": "control_center", "action": "start_curation"},
        )

        # Run curation in background
        self.run_worker(self._run_curation, exclusive=True)
        self.notify("Curation loop started in background")

    async def _run_curation(self) -> None:
        """Run the curation loop as a background worker."""
        import asyncio

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self.curation_tool),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self.call_from_thread(self.notify, "Curation completed successfully!")
                self.call_from_thread(
                    self.emit_bus_event,
                    "strp.control.curation_complete",
                    "metric",
                    {"status": "success"},
                )
            else:
                error_msg = stderr.decode()[:200] if stderr else "Unknown error"
                self.call_from_thread(
                    self.notify,
                    f"Curation failed: {error_msg}",
                    severity="error",
                )
                self.call_from_thread(
                    self.emit_bus_event,
                    "strp.control.curation_complete",
                    "metric",
                    {"status": "error", "error": error_msg},
                    "error",
                )
        except Exception as e:
            self.call_from_thread(
                self.notify,
                f"Curation error: {e}",
                severity="error",
            )
        finally:
            self.curation_running = False
            self.call_from_thread(
                self.update_status,
                "Ready. Press / to enter command, c to curate, w to spawn worker.",
            )

    def action_spawn_worker(self) -> None:
        """Spawn a strp_worker.py process in the background."""
        if not self.worker_tool.exists():
            self.notify("strp_worker.py not found!", severity="error")
            return

        bus_dir = self.pluribus_root / "bus"

        # Build the command to run the worker
        cmd = [
            sys.executable,
            str(self.worker_tool),
            "--bus-dir", str(bus_dir),
            "--provider", "gemini-cli",
        ]

        env = os.environ.copy()
        env["PLURIBUS_BUS_DIR"] = str(bus_dir)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        try:
            # Spawn the worker in the background
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,  # Detach from this process group
            )
            self.spawned_workers.append(proc)

            # Emit event to bus announcing worker spawn
            self.emit_bus_event(
                "strp.control.worker_spawned",
                "request",
                {
                    "source": "control_center",
                    "pid": proc.pid,
                    "provider": "gemini-cli",
                    "bus_dir": str(bus_dir),
                },
            )

            self.notify(f"Worker spawned (PID: {proc.pid})")
            self.update_status(f"Worker spawned with PID {proc.pid}. Waiting for heartbeat...")

        except Exception as e:
            self.notify(f"Failed to spawn worker: {e}", severity="error")

    def action_clear_agent_status(self) -> None:
        """Clear the selected agent's status by emitting a pluribus.check.report event with idle status."""
        if not self.selected_agent:
            self.notify("No agent selected. Click on an agent row first.", severity="warning")
            return

        actor = self.selected_agent

        # Emit a pluribus.check.report event to set the agent to idle
        data = {
            "status": "idle",
            "health": "ok",
            "queue_depth": 0,
            "current_task": {},
            "blockers": [],
            "vor_metrics": {},
            "cleared_by": "control_center",
            "cleared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if self.emit_bus_event(
            "pluribus.check.report",
            "metric",
            data,
            level="info",
            actor=actor,
        ):
            # Update local state immediately
            if actor in self.agent_statuses:
                self.agent_statuses[actor].status = "idle"
                self.agent_statuses[actor].health = "ok"
                self.agent_statuses[actor].queue_depth = 0
                self.agent_statuses[actor].current_task = ""
                self.agent_statuses[actor].blockers = []
                self.refresh_agents()

            self.notify(f"Agent '{actor}' status cleared to idle")
            self.update_status(f"Cleared agent '{actor}' status to idle")
        else:
            self.notify(f"Failed to clear agent '{actor}' status", severity="error")

    def action_view_services(self) -> None:
        """Launch the services management TUI."""
        services_tui = Path(__file__).parent / "services_tui.py"
        if not services_tui.exists():
            self.notify("services_tui.py not found!", severity="error")
            return

        self.update_status("Launching Services TUI...")

        # Run in a new terminal if possible, otherwise in background
        try:
            proc = subprocess.Popen(
                [sys.executable, str(services_tui), "--root", str(self.pluribus_root)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.notify(f"Services TUI launched (PID: {proc.pid})")
            self.update_status("Services TUI launched. Switch to that terminal or press 'v' again.")
        except Exception as e:
            self.notify(f"Failed to launch services TUI: {e}", severity="error")

    def action_view_vps(self) -> None:
        """Show VPS session status in event log panel."""
        self.update_status("Fetching VPS session status...")
        self.load_vps_status()
        log = self.query_one("#event-log", Log)

        # Display VPS status in log panel
        log.write_line("")
        log.write_line("[bold cyan]═══════════════════════════════════════════════════════════[/]")
        log.write_line("[bold cyan]                    VPS SESSION STATUS                      [/]")
        log.write_line("[bold cyan]═══════════════════════════════════════════════════════════[/]")
        log.write_line("")

        # Flow mode
        mode_color = "yellow" if self.vps_flow_mode == "m" else "green"
        mode_name = "Monitor/Approve" if self.vps_flow_mode == "m" else "Automatic"
        log.write_line(f"[bold]Flow Mode:[/] [{mode_color}][{self.vps_flow_mode}] {mode_name}[/]")
        log.write_line("")

        # Providers
        log.write_line("[bold]Providers:[/]")
        for name, status in self.vps_providers.items():
            available = status.get("available", False)
            error = status.get("error", "")
            if available:
                log.write_line(f"  [green][+][/] {name}: Available")
            else:
                error_str = f" ({error})" if error else ""
                log.write_line(f"  [red][-][/] {name}: Unavailable{error_str}")
        log.write_line("")

        # Fallback chain
        log.write_line("[bold]Fallback Chain:[/] codex-cli → gemini → gemini-cli → vertex → mock")
        if self.vps_active_fallback:
            log.write_line(f"[bold]Active:[/] [cyan]{self.vps_active_fallback}[/]")
        log.write_line("")
        log.write_line("[dim]Press 'm' for Monitor mode, 'a' for Auto mode, 'r' to refresh[/]")
        log.write_line("[bold cyan]═══════════════════════════════════════════════════════════[/]")

        self.update_status(f"VPS Mode: [{self.vps_flow_mode}] | Active Fallback: {self.vps_active_fallback or 'none'}")

    def load_vps_status(self) -> None:
        """Load VPS session status from vps_session.py."""
        if not self.vps_tool.exists():
            return

        try:
            result = subprocess.run(
                [sys.executable, str(self.vps_tool), "status", "--json", "--root", str(self.pluribus_root.parent)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse JSON from output (after the text status)
                lines = result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("{"):
                        json_str = "\n".join(lines[i:])
                        try:
                            data = json.loads(json_str)
                            self.vps_flow_mode = data.get("flow_mode", "m")
                            self.vps_providers = data.get("providers", self.vps_providers)
                            self.vps_active_fallback = data.get("active_fallback")
                        except json.JSONDecodeError:
                            pass
                        break
        except Exception:
            pass

    def action_set_mode_monitor(self) -> None:
        """Set VPS flow mode to Monitor/Approve."""
        self.set_vps_flow_mode("m")

    def action_set_mode_auto(self) -> None:
        """Set VPS flow mode to Automatic."""
        self.set_vps_flow_mode("A")

    def set_vps_flow_mode(self, mode: str) -> None:
        """Set VPS flow mode."""
        if not self.vps_tool.exists():
            self.notify("vps_session.py not found!", severity="error")
            return

        try:
            result = subprocess.run(
                [sys.executable, str(self.vps_tool), "mode", mode, "--root", str(self.pluribus_root.parent)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.vps_flow_mode = mode
                mode_name = "Monitor/Approve" if mode == "m" else "Automatic"
                self.notify(f"Flow mode set to [{mode}] {mode_name}")
                self.update_status(f"VPS Mode: [{mode}] {mode_name}")
                # Emit bus event
                self.emit_bus_event("dashboard.vps.flow_mode_changed", "metric", {"mode": mode})
            else:
                self.notify(f"Failed to set flow mode: {result.stderr}", severity="error")
        except Exception as e:
            self.notify(f"Error setting flow mode: {e}", severity="error")

    def action_toggle_dashboard(self) -> None:
        """Toggle dashboard supervisor (start/stop all dashboard services)."""
        if not self.supervisor_tool.exists():
            self.notify("dashboard_supervisor.py not found!", severity="error")
            return

        log = self.query_one("#event-log", Log)

        # Check if supervisor is running
        if self.supervisor_proc and self.supervisor_proc.poll() is None:
            # Stop it
            self.supervisor_proc.terminate()
            try:
                self.supervisor_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.supervisor_proc.kill()
            self.supervisor_proc = None
            log.write_line("[yellow]Dashboard supervisor stopped[/]")
            self.notify("Dashboard supervisor stopped")
            self.update_status("Dashboard: OFF")
        else:
            # Start it
            try:
                self.supervisor_proc = subprocess.Popen(
                    [sys.executable, str(self.supervisor_tool), "daemon"],
                    env={**os.environ, "PLURIBUS_BUS_DIR": str(self.pluribus_root / "bus")},
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                log.write_line(f"[green]Dashboard supervisor started (PID: {self.supervisor_proc.pid})[/]")
                log.write_line("[dim]  Services: VPS Daemon, Dashboard Bridge[/]")
                log.write_line("[dim]  Live reload: enabled[/]")
                self.notify(f"Dashboard supervisor started (PID: {self.supervisor_proc.pid})")
                self.update_status("Dashboard: ON (VPS + Bridge)")
            except Exception as e:
                log.write_line(f"[red]Failed to start dashboard supervisor: {e}[/]")
                self.notify(f"Failed to start: {e}", severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "send-btn":
            self.send_command()
        elif event.button.id == "curation-btn":
            self.action_trigger_curation()
        elif event.button.id == "worker-btn":
            self.action_spawn_worker()
        elif event.button.id == "clear-agent-btn":
            self.action_clear_agent_status()
        elif event.button.id == "services-btn":
            self.action_view_services()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in command input."""
        if event.input.id == "command-input":
            self.send_command()

    def send_command(self) -> None:
        """Send the command from the input field."""
        input_widget = self.query_one("#command-input", Input)
        command = input_widget.value.strip()

        if not command:
            self.notify("No command entered", severity="warning")
            return

        parsed = self.parse_command(command)
        if not parsed:
            self.notify("Invalid command format", severity="error")
            return

        topic, kind, data = parsed

        self.update_status(f"Sending: {topic} ({kind})...")

        if self.emit_bus_event(topic, kind, data):
            self.notify(f"Sent: {topic}")
            input_widget.value = ""
            self.update_status(f"Sent: {topic} ({kind})")
        else:
            self.update_status("Send failed!")

    def on_unmount(self) -> None:
        """Clean up spawned workers on exit."""
        for proc in self.spawned_workers:
            try:
                proc.terminate()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="strp_monitor.py",
        description="STRp Control Center TUI - 'Stir the Pot' - interactive dashboard for Pluribus.",
    )
    p.add_argument(
        "--root",
        default=None,
        help="Path to .pluribus directory (default: search upward from cwd)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.root:
        pluribus_root = Path(args.root).expanduser().resolve()
        # Auto-correct: if user passed repo root but .pluribus exists inside it
        if not pluribus_root.name == ".pluribus" and (pluribus_root / ".pluribus").is_dir():
            pluribus_root = pluribus_root / ".pluribus"
    else:
        pluribus_root = find_pluribus_root(Path.cwd())

    if not pluribus_root or not pluribus_root.exists():
        sys.stderr.write("ERROR: Could not find .pluribus directory.\n")
        sys.stderr.write("Use --root to specify the path.\n")
        return 1

    app = STRpControlCenter(pluribus_root)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
