#!/usr/bin/env python3
"""Pluribus Master TUI.

The Isomorphic Command Center for the Pluribus Agent Mesh.
Parity with Web Dashboard.

Tabs:
1. Dashboard (Metrics, AuOM Sextet)
2. Dialogos (Chat/Command/Interjection)
3. Rhizome (Code/File Browser + Promotion)
4. Services (Registry & Control)
5. Events (Bus Log)
"""
from __future__ import annotations

import sys
import time
import uuid
import os
import json
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import threading

# Import service registry logic
try:
    from service_registry import ServiceRegistry
except ImportError:
    from nucleus.tools.service_registry import ServiceRegistry

# Import plurichat logic
try:
    from plurichat import (
        ChatResponse,
        PLURIBUS_ROOT,
        execute_with_topology,
        get_all_provider_status,
        select_provider_for_query,
        shape_prompt,
        ProviderStatus,
        SuperMotdStatus,
        get_supermotd_status,
        default_actor,
        ChatState, # Now ChatState is part of plurichat.py
        check_browser_daemon_status, # Import browser daemon status check
    )
except ImportError:
    from nucleus.tools.plurichat import (
        ChatResponse,
        PLURIBUS_ROOT,
        execute_with_topology,
        get_all_provider_status,
        select_provider_for_query,
        shape_prompt,
        ProviderStatus,
        SuperMotdStatus,
        get_supermotd_status,
        default_actor,
        ChatState, # Now ChatState is part of plurichat.py
        check_browser_daemon_status, # Import browser daemon status check
    )

sys.dont_write_bytecode = True

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.reactive import reactive
    from textual.screen import ModalScreen, Screen
    from textual.widgets import (
        Button, DataTable, Footer, Header, Static, Input, TabbedContent, TabPane, 
        Log, Tree, DirectoryTree, Label, Markdown
    )
    from textual.binding import Binding
except ImportError:
    sys.stderr.write("ERROR: textual not installed. Run: pip install textual\n")
    sys.exit(1)

from rich import box
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Import service registry logic
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools")) # Adjust path for execution from root
    sys.path.insert(0, str(Path(__file__).parent))
    from service_registry import ServiceRegistry
    from agent_bus import resolve_bus_paths, emit_event
    from semops_lexer import SemopsLexer
except ImportError:
    # Fallback if running from different CWD
    try:
        sys.path.insert(0, ".")
        from nucleus.tools.service_registry import ServiceRegistry
        from nucleus.tools.agent_bus import resolve_bus_paths, emit_event
        from nucleus.tools.semops_lexer import SemopsLexer
    except ImportError as e:
        sys.stderr.write(f"ERROR: required Pluribus modules not importable: {e}\n")
        # Continue without these for UI testing, but disable features
        ServiceRegistry = None

class PlanPreviewModal(ModalScreen):
    """Modal to preview and confirm a repo plan."""
    
    CSS = """
    PlanPreviewModal {
        align: center middle;
    }
    
    .modal-dialog {
        width: 80%;
        height: 80%;
        background: $surface;
        border: solid green;
        padding: 1;
    }
    
    .modal-header {
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 1;
        text-align: center;
    }
    
    #plan-content {
        height: 1fr;
        border: solid gray;
        margin: 1 0;
        overflow-y: scroll;
    }
    
    .modal-buttons {
        height: 3;
        align: right middle;
    }
    
    Button {
        margin-left: 1;
    }
    """
    
    BINDINGS = [("escape", "cancel", "Cancel"), ("c", "confirm", "Confirm")]

    def __init__(self, plan_data: dict, req_id: str, bus_paths: any):
        super().__init__()
        self.plan_data = plan_data
        self.req_id = req_id
        self.bus_paths = bus_paths

    def compose(self) -> ComposeResult:
        preview = self.plan_data.get("preview_snippet", "No preview available.")
        target = self.plan_data.get("target_path", "unknown")
        steps = self.plan_data.get("plan_steps", [])
        
        md_text = f"**Target:** `{target}`\n\n**Steps:**\n"
        for i, step in enumerate(steps):
            md_text += f"{i+1}. **{step.get('step', '?')}**: {step.get('description', '')}\n"
            
        md_text += f"\n**Preview:**\n```\n{preview}\n```"

        with Container(classes="modal-dialog"):
            yield Label("🧬 Ribosome Translation Plan", classes="modal-header")
            yield Markdown(md_text, id="plan-content")
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel (Esc)", variant="error", id="btn-cancel")
                yield Button("Manifest (c)", variant="success", id="btn-confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-confirm":
            self.action_confirm()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def action_confirm(self) -> None:
        # Emit exec request
        emit_event(
            self.bus_paths,
            topic="repo.exec.request",
            kind="request",
            level="info",
            actor="tui-user",
            data={
                "req_id": self.req_id,
                "plan_ref": self.plan_data.get("source_sha", "unknown"),
                "stage_path": self.plan_data.get("target_path"),
                "checks": ["lint", "test"]
            },
            trace_id=None,
            run_id=None,
            durable=True
        )
        self.dismiss(result=True)

    def action_cancel(self) -> None:
        self.dismiss(result=False)


class DynamicHeader(Static):
    """A living header that responds to the Art Director."""
    
    DEFAULT_CSS = """
    DynamicHeader {
        height: 8;
        dock: top;
        content-align: center middle;
        text-style: bold;
        color: $text;
        background: $surface;
        border-bottom: solid $primary;
    }
    """
    
    entropy = reactive(0.0)
    current_composition = reactive({})

    def on_mount(self) -> None:
        self.update_art("calm", 0.0, {})
        self.set_interval(5.0, self.tick) # Animate slightly

    def tick(self) -> None:
        # Subtle breathing or glitch if high entropy
        if self.entropy > 0.3:
            self.update_art(self.app.current_mood, self.entropy, self.app.current_composition) # Re-render with current state

    def update_art(self, mood: str, entropy: float, composition: dict) -> None:
        import random
        # Pick a base genome
        raw = random.choice(self.app.ASCII_GENOME) if hasattr(self.app, "ASCII_GENOME") else "PLURIBUS"
        
        # Glitch it?
        if entropy > 0.3:
            chars = list(raw)
            for i in range(len(chars)):
                if chars[i].strip() and random.random() < (entropy * 0.1):
                    chars[i] = random.choice(["*", "#", "@", "?", "!", "Ω", "µ", "ß"])
            raw = "".join(chars)
            
        # Construct Debugging Footer
        comp_id = composition.get("id", "N/A")
        shader_id = composition.get("shaderId", "N/A")
        
        tokens = composition.get("tokens", {})
        primary_color = tokens.get("--art-primary", "N/A")
        blur_px = tokens.get("blur_px", "N/A")
        
        footer = (
            f"[dim]COMP ID:[/] [magenta]{comp_id}[/] | "
            f"[dim]SHADER:[/] [cyan]{shader_id}[/] | "
            f"[dim]MOOD:[/] [yellow]{mood}[/] | "
            f"[dim]ENTROPY:[/] [red]{entropy:.2f}[/] | "
            f"[dim]ANXIETY:[/] [red]{self.app.anxiety:.2f}[/]\n" # Access directly from app as reactive
            f"[dim]PRIMARY:[/] [orange3]{primary_color}[/] | [dim]BLUR:[/] [orange3]{blur_px}[/] "
        )
        
        self.update(f"{raw}\n\n{footer}")
        
        # Style mapping
        colors = {
            "calm": "green", "anxious": "red", "hyper": "cyan", 
            "chaotic": "magenta", "dormant": "grey", "focused": "blue"
        }
        c = colors.get(mood, "white")
        self.styles.color = c
        self.styles.border_bottom = ("solid", c)

class GestaltBar(Static):
    """Compact gestalt status bar showing generation/build/lineage info."""

    DEFAULT_CSS = """
    GestaltBar {
        height: 1;
        dock: top;
        text-style: dim;
        color: $text-muted;
        background: $surface-darken-1;
        padding: 0 1;
    }
    """

    generation = reactive(None)
    lineage_id = reactive(None)
    dag_id = reactive(None)

    def on_mount(self) -> None:
        self.refresh_lineage()
        self.set_interval(30.0, self.refresh_lineage)
        self.set_interval(2.0, self.render_bar)

    def refresh_lineage(self) -> None:
        import json
        lineage_candidates = [
            Path("/pluribus/.pluribus/lineage.json"),
            Path("/pluribus/.pluribus_local/lineage.json"),
        ]
        for lineage_path in lineage_candidates:
            if not lineage_path.exists():
                continue
            try:
                data = json.loads(lineage_path.read_text(encoding="utf-8"))
                self.generation = data.get("generation")
                self.lineage_id = (data.get("lineage_id", "") or "")[:8]
                self.dag_id = (data.get("dag_id", "") or "")[:8]
                break
            except Exception:
                continue
        self.render_bar()

    def render_bar(self) -> None:
        gen = self.generation if self.generation is not None else "—"
        lin = self.lineage_id or "—"
        dag = self.dag_id or "—"
        started_at = getattr(self.app, "started_at", None)
        build_time = started_at.strftime("%H:%M") if started_at else datetime.now().strftime("%H:%M")

        mood = getattr(self.app, "current_mood", None) or "—"
        mood_colors = {
            "calm": "green",
            "anxious": "red",
            "hyper": "cyan",
            "chaotic": "magenta",
            "dormant": "grey",
            "focused": "blue",
        }
        dot_color = mood_colors.get(str(mood), "white")
        dot = f"[{dot_color}]●[/{dot_color}]"

        composition = getattr(self.app, "current_composition", None) or {}
        comp_id = (composition.get("id") or composition.get("name") or "—")
        comp_id = str(comp_id)
        if len(comp_id) > 14:
            comp_id = comp_id[:14] + "…"

        bar = (
            f"[cyan]📄[/cyan] "
            f"[bold]GEN:{gen}[/bold] | "
            f"[dim]{build_time}[/dim] | "
            f"{dot} [yellow]{mood}[/yellow] | "
            f"[dim]COMP:[/dim] [magenta]{comp_id}[/magenta] | "
            f"[dim]LIN:{lin}[/dim] | "
            f"[dim]DAG:{dag}[/dim]"
        )
        self.update(bar)


class HGTInputModal(ModalScreen):
    """Modal to input HGT Source SHA."""
    CSS = """
    HGTInputModal {
        align: center middle;
    }
    .modal-dialog {
        width: 60%;
        height: auto;
        background: $surface;
        border: solid pink;
        padding: 1;
    }
    .modal-header {
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 1;
        text-align: center;
    }
    Input {
        margin: 1 0;
    }
    .modal-buttons {
        height: 3;
        align: right middle;
    }
    Button {
        margin-left: 1;
    }
    """
    
    BINDINGS = [("escape", "cancel", "Cancel"), ("enter", "confirm", "Confirm")]

    def compose(self) -> ComposeResult:
        with Container(classes="modal-dialog"):
            yield Label("🧬 Horizontal Gene Transfer (HGT)", classes="modal-header")
            yield Label("Enter Source SHA / Ref to Splice:")
            yield Input(placeholder="e.g. origin/feature or <sha>", id="hgt-sha")
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", variant="error", id="btn-cancel")
                yield Button("Splice", variant="success", id="btn-confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-confirm":
            self.action_confirm()
        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def action_confirm(self) -> None:
        val = self.query_one("#hgt-sha", Input).value
        if val:
            self.dismiss(val)
        else:
            self.notify("SHA required", severity="error")

class DiffModal(ModalScreen):
    """Modal to view Diff/Commit details."""
    CSS = """
    DiffModal {
        align: center middle;
    }
    .modal-dialog {
        width: 90%;
        height: 90%;
        background: $surface;
        border: solid cyan;
        padding: 1;
    }
    #diff-content {
        height: 1fr;
        border: solid gray;
        overflow-y: scroll;
    }
    """
    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, content: str, title: str = "Diff"):
        super().__init__()
        self.content = content
        self.title_text = title

    def compose(self) -> ComposeResult:
        with Container(classes="modal-dialog"):
            yield Label(self.title_text, classes="modal-header")
            yield Markdown(self.content, id="diff-content")
            yield Button("Close (Esc)", variant="default", id="btn-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()


class TopologyWidget(Static):
    """Visualizes the current agent topology state."""

    DEFAULT_CSS = """
    TopologyWidget {
        height: 10;
        border: solid yellow;
        padding: 1;
    }
    .topo-header {
        text-style: bold;
        color: yellow;
        dock: top;
    }
    .topo-value {
        text-align: center;
        text-style: bold;
        color: white;
        margin-top: 1;
    }
    .topo-detail {
        text-align: center;
        color: gray;
    }
    """

    topology = reactive("single")
    fanout = reactive(1)
    reason = reactive("default")

    def compose(self) -> ComposeResult:
        yield Label("Active Topology", classes="topo-header")
        yield Label(f"{self.topology.upper()}", classes="topo-value", id="topo-main")
        yield Label(f"Fanout: {self.fanout}", classes="topo-detail", id="topo-fanout")
        yield Label(f"Reason: {self.reason}", classes="topo-detail", id="topo-reason")

    def watch_topology(self, val: str) -> None:
        try:
            self.query_one("#topo-main", Label).update(val.upper())
            color = "green" if val == "single" else "cyan" if val == "star" else "magenta"
            self.styles.border = ("solid", color)
            self.query_one(".topo-header").styles.color = color
        except Exception:
            pass

    def watch_fanout(self, val: int) -> None:
        try:
            self.query_one("#topo-fanout", Label).update(f"Fanout: {val}")
        except Exception:
            pass

    def watch_reason(self, val: str) -> None:
        try:
            self.query_one("#topo-reason", Label).update(f"Reason: {val}")
        except Exception:
            pass


def _clamp(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def _format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = minutes / 60.0
    if hours < 24:
        return f"{int(hours)}h"
    days = hours / 24.0
    return f"{int(days)}d"


def _parse_iso_ts(iso: str) -> float | None:
    if not iso:
        return None
    try:
        return time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return None


def _tail_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    chunk_size = 8192
    data = b""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            while end > 0 and data.count(b"\n") <= max_lines:
                start = max(0, end - chunk_size)
                handle.seek(start)
                data = handle.read(end - start) + data
                end = start
    except Exception:
        return []
    lines = [ln.decode("utf-8", errors="ignore") for ln in data.splitlines() if ln.strip()]
    return lines[-max_lines:]


def _tail_ndjson(path: Path, max_lines: int) -> list[dict]:
    lines = _tail_lines(path, max_lines)
    out: list[dict] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


class MotdScreen(Screen):
    CSS = """
    MotdScreen {
        layout: vertical;
    }
    #motd-header {
        height: 3;
        padding: 0 1;
        background: $surface;
        color: $text;
        border-bottom: solid $primary;
    }
    #motd-title {
        text-style: bold;
    }
    #motd-tabs {
        height: 1fr;
    }
    #motd-yt-scroll, #motd-semops-scroll {
        height: 1fr;
        padding: 1;
    }
    #motd-pulse-grid {
        layout: grid;
        grid-size: 3 3;
        grid-columns: 1fr 1fr 1fr;
        grid-rows: 1fr 1fr auto;
        gap: 1;
        padding: 1;
        height: 1fr;
    }
    #motd-bus {
        height: 1fr;
    }
    #motd-agents {
        height: 1fr;
    }
    #motd-services {
        height: 1fr;
    }
    #motd-tasks-scroll {
        height: 1fr;
        grid-column: 1 / span 3;
        grid-row: 2;
    }
    #motd-resources {
        height: 4;
        grid-column: 1 / span 3;
        grid-row: 3;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("left", "prev_tab", show=False),
        Binding("right", "next_tab", show=False),
        Binding("tab", "next_tab", show=False),
        Binding("shift+tab", "prev_tab", show=False),
    ]

    def __init__(
        self,
        root: Path,
        bus_dir: Path | None = None,
        state_root: Path | None = None,
        semops_path: Path | None = None,
        registry: ServiceRegistry | None = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.bus_dir = bus_dir
        self.state_root = state_root or (bus_dir.parent if bus_dir else root / ".pluribus")
        self.semops_path = semops_path or (root / "nucleus" / "specs" / "semops.json")
        self.registry = registry
        self._bus_events: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Container(
            Static(
                "MOTD: Tab/Left/Right to switch views | Esc back | Esc Esc exit",
                id="motd-title",
            ),
            id="motd-header",
        )
        with TabbedContent(id="motd-tabs"):
            with TabPane("YouTube", id="motd-yt"):
                yield ScrollableContainer(Static(id="motd-yt-content"), id="motd-yt-scroll")
            with TabPane("SemOps", id="motd-semops"):
                yield ScrollableContainer(Static(id="motd-semops-content"), id="motd-semops-scroll")
            with TabPane("Pulse", id="motd-pulse"):
                yield Container(
                    Static(id="motd-bus"),
                    Static(id="motd-agents"),
                    Static(id="motd-services"),
                    ScrollableContainer(Static(id="motd-tasks"), id="motd-tasks-scroll"),
                    Static(id="motd-resources"),
                    id="motd-pulse-grid",
                )

    def on_mount(self) -> None:
        self.refresh_youtube()
        self.refresh_semops()
        self.refresh_pulse()
        self.set_interval(2.0, self.refresh_pulse)
        self.set_interval(15.0, self.refresh_youtube)
        self.set_interval(20.0, self.refresh_semops)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_next_tab(self) -> None:
        self._cycle_tab(1)

    def action_prev_tab(self) -> None:
        self._cycle_tab(-1)

    def _cycle_tab(self, delta: int) -> None:
        try:
            tabs = self.query_one("#motd-tabs", TabbedContent)
            panes = list(tabs.query(TabPane))
            pane_ids = [pane.id for pane in panes if pane.id]
            if not pane_ids:
                return
            current = tabs.active
            if current not in pane_ids:
                tabs.active = pane_ids[0]
                return
            idx = pane_ids.index(current)
            tabs.active = pane_ids[(idx + delta) % len(pane_ids)]
        except Exception:
            pass

    def _events_path(self) -> Path | None:
        if self.bus_dir:
            return self.bus_dir / "events.ndjson"
        candidate = self.root / ".pluribus" / "bus" / "events.ndjson"
        if candidate.exists():
            return candidate
        fallback = self.root / ".pluribus_local" / "bus" / "events.ndjson"
        return fallback if fallback.exists() else None

    def _task_ledger_path(self) -> Path | None:
        primary = self.state_root / "index" / "task_ledger.ndjson"
        if primary.exists():
            return primary
        fallback = self.root / ".pluribus_local" / "index" / "task_ledger.ndjson"
        return fallback if fallback.exists() else None

    def _event_ts(self, evt: dict) -> float:
        ts = evt.get("ts")
        if isinstance(ts, (int, float)):
            return float(ts)
        iso = evt.get("iso") or evt.get("data", {}).get("iso")
        parsed = _parse_iso_ts(str(iso)) if iso else None
        return parsed or 0.0

    def refresh_pulse(self) -> None:
        events_path = self._events_path()
        if events_path:
            self._bus_events = _tail_ndjson(events_path, 240)
        else:
            self._bus_events = []
        self._update_bus_panel()
        self._update_agents_panel()
        self._update_services_panel()
        self._update_tasks_panel()
        self._update_resources_panel()

    def refresh_youtube(self) -> None:
        cache_dir = Path(os.environ.get("MOTD_CURATE_CACHE_DIR", "/tmp"))
        prefix = os.environ.get("MOTD_CURATE_PREFIX", "ingest_thumbs")
        meta_path = cache_dir / f"{prefix}_meta.json"
        config_path = Path(
            os.environ.get(
                "MOTD_CURATE_CONFIG",
                str(Path.home() / ".config" / "motd-curate" / "sources.json"),
            )
        )

        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        topics: list[str] = []
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                topics_cfg = cfg.get("topics")
                if isinstance(topics_cfg, dict):
                    topics.extend([str(k) for k in topics_cfg.keys()])
                elif isinstance(topics_cfg, list):
                    for item in topics_cfg:
                        if isinstance(item, dict):
                            name = item.get("name") or item.get("topic")
                            if name:
                                topics.append(str(name))
                        elif isinstance(item, str):
                            topics.append(item)
            except Exception:
                topics = []

        table = Table(box=box.SIMPLE, expand=True, show_header=True)
        table.add_column("Slot", width=4)
        table.add_column("Title", ratio=3)
        table.add_column("Channel", ratio=2)
        table.add_column("Dur", width=6)
        table.add_column("Topic", ratio=2)

        items = meta.get("items") if isinstance(meta.get("items"), list) else []
        if items:
            for item in items:
                if not isinstance(item, dict):
                    continue
                slot = str(item.get("slot", "?"))
                title = _clamp(str(item.get("title") or "Untitled"), 48)
                channel = _clamp(str(item.get("channel") or item.get("uploader") or ""), 24)
                duration = _clamp(str(item.get("duration") or ""), 6)
                topic = _clamp(str(item.get("topic") or ""), 18)
                table.add_row(slot, title, channel, duration, topic)
        else:
            table.add_row("-", "No cached picks (run motd-curate update)", "", "", "")

        topic_lines = topics[:18] if topics else ["No topics configured"]
        topics_text = Text("Topics: " + ", ".join(topic_lines), style="dim")
        generated = meta.get("generated_at")
        generated_line = ""
        if isinstance(generated, (int, float)):
            generated_line = f"Cache: {time.strftime('%Y-%m-%d %H:%M', time.localtime(float(generated)))}"
        elif isinstance(generated, str):
            generated_line = f"Cache: {generated}"
        if generated_line:
            topics_text.append("\n" + generated_line, style="dim")

        content = Group(table, topics_text)
        try:
            self.query_one("#motd-yt-content", Static).update(content)
        except Exception:
            pass

    def refresh_semops(self) -> None:
        ops = {}
        if self.semops_path.exists():
            try:
                data = json.loads(self.semops_path.read_text(encoding="utf-8"))
                ops = data.get("operators") or {}
            except Exception:
                ops = {}

        table = Table(box=box.SIMPLE, expand=True, show_header=True)
        table.add_column("KEY", width=8)
        table.add_column("Domain", width=12)
        table.add_column("Topic", width=20)
        table.add_column("Summary", ratio=3)

        if isinstance(ops, dict) and ops:
            for key in sorted(ops.keys())[:28]:
                op = ops.get(key, {})
                domain = _clamp(str(op.get("domain") or ""), 12)
                topic = _clamp(str(op.get("bus_topic") or ""), 24)
                desc = _clamp(str(op.get("description") or ""), 64)
                table.add_row(key, domain, topic, desc)
        else:
            table.add_row("-", "No SemOps loaded", "", "")

        info = Text("PB SemOps cheat sheet (live from semops.json)", style="dim")
        content = Group(info, table)
        try:
            self.query_one("#motd-semops-content", Static).update(content)
        except Exception:
            pass

    def _update_bus_panel(self) -> None:
        events = self._bus_events[-14:]
        table = Table(box=box.SIMPLE, expand=True, show_header=True)
        table.add_column("Time", width=8)
        table.add_column("Topic", ratio=3)
        table.add_column("Actor", ratio=1)
        if events:
            for evt in events:
                iso = str(evt.get("iso") or "")[11:19]
                if not iso:
                    iso = _format_age(time.time() - self._event_ts(evt))
                topic = _clamp(str(evt.get("topic") or ""), 36)
                actor = _clamp(str(evt.get("actor") or ""), 14)
                table.add_row(iso, topic, actor)
        else:
            table.add_row("--", "No bus activity", "")
        panel = Panel(table, title="BUS (live)", border_style="cyan")
        try:
            self.query_one("#motd-bus", Static).update(panel)
        except Exception:
            pass

    def _update_agents_panel(self) -> None:
        counts: dict[str, int] = {}
        last_seen: dict[str, float] = {}
        for evt in self._bus_events:
            actor = str(evt.get("actor") or "unknown")
            counts[actor] = counts.get(actor, 0) + 1
            last_seen[actor] = max(last_seen.get(actor, 0.0), self._event_ts(evt))

        table = Table(box=box.SIMPLE, expand=True, show_header=True)
        table.add_column("Agent", ratio=2)
        table.add_column("Last", width=6)
        table.add_column("Cnt", width=4)

        if counts:
            for actor in sorted(last_seen, key=lambda a: last_seen[a], reverse=True)[:12]:
                age = _format_age(time.time() - last_seen[actor])
                table.add_row(_clamp(actor, 18), age, str(counts.get(actor, 0)))
        else:
            table.add_row("--", "--", "0")

        panel = Panel(table, title="AGENTS", border_style="green")
        try:
            self.query_one("#motd-agents", Static).update(panel)
        except Exception:
            pass

    def _update_services_panel(self) -> None:
        table = Table(box=box.SIMPLE, expand=True, show_header=True)
        table.add_column("Service", ratio=2)
        table.add_column("Status", ratio=1)
        table.add_column("PID", width=6)

        if self.registry:
            try:
                self.registry.load()
                svc_map = {s.id: {"def": s, "inst": None} for s in self.registry.list_services()}
                for inst in self.registry.list_instances():
                    if inst.service_id in svc_map:
                        svc_map[inst.service_id]["inst"] = inst
                ordered = sorted(
                    svc_map.items(),
                    key=lambda item: 0
                    if item[1]["inst"] and item[1]["inst"].status == "running"
                    else 1,
                )
                for sid, info in ordered[:16]:
                    inst = info["inst"]
                    status = "stopped"
                    pid = "-"
                    if inst and inst.status == "running":
                        status = "running"
                        pid = str(inst.pid)
                    elif inst and inst.status == "error":
                        status = "error"
                    table.add_row(_clamp(sid, 20), status, pid)
            except Exception:
                table.add_row("--", "registry error", "--")
        else:
            table.add_row("--", "no registry", "--")

        panel = Panel(table, title="SERVICES", border_style="blue")
        try:
            self.query_one("#motd-services", Static).update(panel)
        except Exception:
            pass

    def _update_tasks_panel(self) -> None:
        ledger_path = self._task_ledger_path()
        entries = _tail_ndjson(ledger_path, 400) if ledger_path else []
        tasks: dict[tuple[str, str], dict] = {}
        for entry in entries:
            run_id = entry.get("run_id") or entry.get("meta", {}).get("run_id")
            if not run_id:
                continue
            actor = str(entry.get("actor") or "unknown")
            ts = entry.get("ts")
            if not isinstance(ts, (int, float)):
                ts = _parse_iso_ts(str(entry.get("iso") or "")) or 0.0
            key = (actor, str(run_id))
            if key not in tasks or ts >= tasks[key].get("_ts", 0.0):
                entry["_ts"] = ts
                tasks[key] = entry

        bus_updates: dict[tuple[str, str], dict] = {}
        for evt in self._bus_events:
            topic = str(evt.get("topic") or "")
            if not any(x in topic for x in ("task", "strp", "dialogos", "infercell")):
                continue
            data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
            req_id = data.get("req_id") or data.get("run_id") or evt.get("run_id")
            if not req_id:
                continue
            actor = str(evt.get("actor") or "unknown")
            ts = self._event_ts(evt)
            key = (actor, str(req_id))
            if key not in bus_updates or ts >= bus_updates[key].get("_ts", 0.0):
                bus_updates[key] = {"_ts": ts, "topic": topic}

        for key, info in bus_updates.items():
            if key not in tasks:
                tasks[key] = {
                    "actor": key[0],
                    "run_id": key[1],
                    "status": "bus",
                    "meta": {"desc": info.get("topic")},
                    "_ts": info.get("_ts", 0.0),
                }

        by_actor: dict[str, list[dict]] = {}
        for (actor, _), entry in tasks.items():
            by_actor.setdefault(actor, []).append(entry)

        panels = []
        now = time.time()
        for actor in sorted(by_actor.keys())[:6]:
            entries = sorted(by_actor[actor], key=lambda e: e.get("_ts", 0.0), reverse=True)
            lines = []
            for entry in entries[:6]:
                run_id = str(entry.get("run_id") or "")[:8]
                status = str(entry.get("status") or "event")
                ts = float(entry.get("_ts") or 0.0)
                for key in ((actor, str(entry.get("run_id") or "")),):
                    if key in bus_updates:
                        ts = max(ts, bus_updates[key].get("_ts", ts))
                age = now - ts if ts else 0.0
                age_label = _format_age(age) if age else "--"
                state = "ip" if status == "in_progress" else status[:8]
                if status == "in_progress" and age > 900:
                    state = "stalled"
                meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
                desc = meta.get("desc") or meta.get("step") or entry.get("topic") or ""
                desc = _clamp(str(desc), 28)
                line = f"{run_id:8} {state:8} {age_label:>4} {desc}"
                lines.append(line)
            if not lines:
                lines = ["no tasks"]
            body = Text("\n".join(lines))
            panels.append(Panel(body, title=_clamp(actor, 18), border_style="magenta"))

        repo_path = os.environ.get("PLURIBUS_TRILATERREPO") or str(self.root)
        header = Text(f"trilaterrepo: {repo_path}", style="dim")
        content = Group(header, Columns(panels, expand=True, equal=True) if panels else Text("No tasks", style="dim"))
        panel = Panel(content, title="TASKS (live)", border_style="magenta")
        try:
            self.query_one("#motd-tasks", Static).update(panel)
        except Exception:
            pass

    def _render_resource_meter(self, label: str, used: int, total: int, free: int) -> Text:
        pct = (used / total * 100.0) if total else 0.0
        color = "green" if pct < 70 else "yellow" if pct < 85 else "red"
        filled = int((pct / 100.0) * 12)
        filled = max(0, min(12, filled))
        bar = Text()
        bar.append("[", style="dim")
        bar.append("#" * filled, style=color)
        bar.append("-" * (12 - filled), style="dim")
        bar.append("]", style="dim")

        text = Text()
        text.append(f"{label} {pct:>4.0f}%\n", style="cyan")
        text.append_text(bar)
        text.append(
            f" { _format_bytes(used) }/{ _format_bytes(total) } free { _format_bytes(free) }",
            style="dim",
        )
        return text

    def _update_resources_panel(self) -> None:
        candidates = [
            Path("/"),
            self.root,
            Path("/pluribus"),
            Path("/var/lib/pluribus"),
            Path("/tmp"),
        ]
        if self.bus_dir:
            candidates.append(self.bus_dir)

        meters = []
        seen_paths: set[str] = set()
        for path in candidates:
            if not path.exists():
                continue
            try:
                label = str(path)
                if label in seen_paths:
                    continue
                seen_paths.add(label)
                usage = shutil.disk_usage(path)
                label = _clamp(label, 14)
                meters.append(self._render_resource_meter(label, usage.used, usage.total, usage.free))
            except Exception:
                continue
        if len(meters) > 5:
            meters = meters[:5]

        if not meters:
            content = Text("No storage metrics available", style="dim")
        else:
            content = Columns(meters, expand=True, equal=True)
        panel = Panel(content, title="RESOURCES", border_style="yellow")
        try:
            self.query_one("#motd-resources", Static).update(panel)
        except Exception:
            pass

class PluribusTUI(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 1 2;
        grid-rows: 1fr auto;
    }
    
    .box {
        height: 100%;
        border: solid green;
    }
    
    #metrics-grid {
        layout: grid;
        grid-size: 2 2;
        height: auto;
        margin-bottom: 1;
    }
    
    .metric-card {
        border: solid cyan;
        height: 10;
        padding: 1;
    }
    
    .card-title {
        text-style: bold;
        color: cyan;
        dock: top;
    }
    
    .card-value {
        text-align: center;
        text-style: bold;
        color: green;
        margin-top: 2;
    }

    #dialogos-output {
        height: 1fr;
        border: solid magenta;
        overflow-y: scroll;
    }
    
    #rhizome-artifacts {
        width: 1fr;
        height: 1fr;
    }
    .rhizome-left-pane {
        width: 40%;
        height: 100%;
        dock: left;
        border-right: solid cyan;
    }
    .rhizome-right-pane {
        width: 60%;
        height: 100%;
        overflow-y: scroll;
        padding: 1;
    }
    .button-row {
        height: 3;
        padding: 0 1;
    }

    /* Plurichat Tab Specific Styles */
    .plurichat-main-pane {
        width: 65%;
        height: 100%;
        border-right: solid $primary;
    }
    .plurichat-side-pane {
        width: 35%;
        height: 100%;
    }
    .plurichat-transcript-pane {
        height: 70%;
    }
    .plurichat-input-box {
        height: 3;
        border-top: solid $primary;
        color: $primary;
    }
    .plurichat-controls-top {
        height: 3;
        padding: 0 1;
        background: $surface;
        color: $text;
        border-bottom: solid $primary;
    }
    .plurichat-control-label {
        width: auto;
        text-style: dim;
        margin-right: 1;
    }
    .plurichat-control-value {
        width: auto;
        text-style: bold;
        color: cyan;
        margin-right: 2;
    }
    .plurichat-status-pane {
        height: 30%;
        border-bottom: solid $primary;
    }
    .plurichat-agent-pane {
        height: 70%;
    }

    #code-view {
        width: 70%;
        height: 100%;
        overflow-y: scroll;
    }

    .muse-status {
        dock: top;
        width: auto;
        padding: 0 1;
        background: $surface;
        color: $text;
        text-style: bold;
        text-align: right;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("p", "promote", "Promote Artifact", show=True, key_display="p"),
        Binding("escape", "escape", "Back", show=False),
    ]

    TITLE = "Pluribus Master Control"
    
    # Reactives for Metrics
    vor_cdi = reactive(0.0)
    omega_beats = reactive(0)
    bus_events = reactive(0)
    sextet_balance = reactive("Balanced")
    
    # Art Dept Integration
    current_mood = reactive("calm")
    entropy = reactive(0.0)
    anxiety = reactive(0.0)
    current_composition = reactive({})
    
    # UI State
    view_mode = reactive("markdown")
    search_query = reactive("")
    chat_input_buf = reactive("")

    # Plurichat State
    chat_state = reactive(None) # type: ChatState
    
    def __init__(self, root: Path, bus_dir: Path | None = None, motd_mode: bool = False):
        super().__init__()
        self.started_at = datetime.now()
        self.root = root
        self.motd_mode = motd_mode
        self._esc_armed = False
        if resolve_bus_paths:
            self.bus_paths = resolve_bus_paths(str(bus_dir) if bus_dir else None)
            self.bus_dir = Path(self.bus_paths.active_dir)
            self.state_root = self.bus_dir.parent
            self.ARTIFACTS_NDJSON_PATH = self.state_root / "index" / "artifacts.ndjson"
            self.RHIZOME_CONTENT_PATH = self.state_root / "objects"
            self.SEMOPS_PATH = root / "nucleus" / "specs" / "semops.json"
            self.USER_OPS_PATH = self.state_root / "user_operators.json"
        else:
            self.bus_paths = None
            self.state_root = root
        
        if ServiceRegistry:
            self.registry = ServiceRegistry(root)
        else:
            self.registry = None

        # State for Promotion Flow
        self.active_plan_req_id = None
        self.last_bus_offset = 0
        self.chat_state = ChatState(bus_dir=self.bus_dir, actor=default_actor())

    ASCII_GENOME = [
        r"""
  ____  _       _   _     _ _ _                 
 |  _ \| |     | | (_)   | (_) |                
 | |_) | |_   _| |_ _  __| |_| |_ ___  _ __ ___ 
 |  _ <| | | | | __| |/ _` | | __/ _ \| '__/ __|
 | |_) | | |_| | |_| | (_| | | || (_) | |  \__ \\
 |____/|_|\__,_|\__|_|\__,_|_|\__\___/|_|  |___/
        """,
        r"""
 █▀█ █   █ █ █▀█ ▀█▀ █▀▄ █ █ █▀
 █▀▀ █▄▄ █▄█ █▀▄ ▄█▄ █▄▀ █▄█ ▄█
        """,
        r"""
 .--.  .    .  .---..---..---..---. .  . .---.
 |   ) |    |  |    |    |    |   | |  | |    
 |---' |    |  |--- |--- |--- |---' |  | `---.
 |     |    |  |    |    |    |  \  |  |     |
 '     '---'`--'    '    '---''   ` `  ` '---'
        """
    ]

    def watch_current_mood(self, mood: str) -> None:
        """React to mood changes by updating theme colors."""
        colors = {
            "calm": "green",
            "anxious": "red",
            "hyper": "cyan",
            "chaotic": "magenta",
            "dormant": "grey",
            "focused": "blue",
        }
        color = colors.get(mood, "white")
        
        # Update styles dynamically
        self.screen.styles.border = ("solid", color)
        for widget in self.query(".metric-card"):
            widget.styles.border = ("solid", color)
        
        # Trigger header update
        try:
            self.query_one(DynamicHeader).update_art(mood, self.entropy, self.current_composition)
        except: pass

    def watch_entropy(self, val: float) -> None:
        """Entropy drives glitch intensity."""
        try:
            self.query_one(DynamicHeader).entropy = val
            # Also trigger re-render on entropy change
            self.query_one(DynamicHeader).update_art(self.current_mood, val, self.current_composition)
        except: pass

    def compose(self) -> ComposeResult:
        yield DynamicHeader(id="dyn-header")
        yield GestaltBar(id="gestalt-bar")
        yield Label(f"Ω Muse: {self.current_mood}", id="muse-status", classes="muse-status")
        
        with TabbedContent():
            # 1. Dashboard
            with TabPane("Dashboard", id="tab-dash"):
                yield Container(
                    Static("System Metrics", classes="panel-title"),
                    Container(
                        Vertical(
                            Static("VOR CDI", classes="card-title"),
                            Label(f"{self.vor_cdi:.2f}", classes="card-value", id="val-cdi"),
                            classes="metric-card"
                        ),
                        Vertical(
                            Static("ω-Automata", classes="card-title"),
                            Label(f"{self.omega_beats}", classes="card-value", id="val-omega"),
                            classes="metric-card"
                        ),
                        Vertical(
                            Static("Bus Events", classes="card-title"),
                            Label(f"{self.bus_events}", classes="card-value", id="val-bus"),
                            classes="metric-card"
                        ),
                        Vertical(
                            Static("AuOM Sextet", classes="card-title"),
                            Label(f"{self.sextet_balance}", classes="card-value", id="val-sextet"),
                            classes="metric-card"
                        ),
                        TopologyWidget(id="topology-widget"),
                        id="metrics-grid"
                    ),
                    Button("Force Muse (Trigger Art Director)", id="btn-force-muse", variant="primary")
                )

            # 2. Dialogos
            with TabPane("Dialogos", id="tab-chat"):
                yield Vertical(
                    Log(id="dialogos-log", highlight=True),
                    Input(placeholder="Interject command (enter to send)...", id="cmd-input"),
                )

            # 3. Plurichat Harness
            with TabPane("Plurichat", id="tab-plurichat"):
                with Horizontal():
                    # Left Pane: Chat Transcript and Input
                    yield Vertical(
                        ScrollableContainer(
                            Log(id="plurichat-transcript", highlight=True),
                            classes="plurichat-transcript-pane"
                        ),
                        Input(placeholder="Message (Enter to send)...", id="plurichat-input", classes="plurichat-input-box"),
                        classes="plurichat-main-pane"
                    )
                    
                    # Right Pane: Controls, Status, Agent Activity
                    yield Vertical(
                        # Top: Quick Controls (Provider, Persona)
                        Horizontal(
                            Static("Provider:", classes="plurichat-control-label"),
                            Label("auto", id="plurichat-provider-label", classes="plurichat-control-value"),
                            Static("Persona:", classes="plurichat-control-label"),
                            Label("auto", id="plurichat-persona-label", classes="plurichat-control-value"),
                            Static("Context:", classes="plurichat-control-label"),
                            Label("auto", id="plurichat-context-label", classes="plurichat-control-value"),
                            classes="plurichat-controls-top"
                        ),
                        # Middle: SUPERMOTD / Omega / Provider Status
                        ScrollableContainer(
                            Markdown("Loading SUPERMOTD...", id="plurichat-supermotd"),
                            classes="plurichat-status-pane"
                        ),
                        # Bottom: InferCell / Agent Topology / Browser Daemon
                        ScrollableContainer(
                            Markdown("Loading agent activity...", id="plurichat-agent-activity"),
                            classes="plurichat-agent-pane"
                        ),
                        classes="plurichat-side-pane"
                    )

            # 4. Voice (AURALUX)
            with TabPane("Voice", id="tab-voice"):
                yield Vertical(
                    Label("AURALUX Voice Console", classes="panel-title"),
                    Markdown("STT/TTS Pipeline Status: [green]READY[/green]\n\n**Latency:** 120ms\n**VAD:** Idle\n**Active Profile:** Default", id="voice-status"),
                    Horizontal(
                        Button("Start Mic", id="btn-voice-start", variant="success"),
                        Button("Stop Mic", id="btn-voice-stop", variant="error"),
                        Button("Test TTS", id="btn-voice-tts", variant="primary"),
                        classes="button-row"
                    ),
                    Log(id="voice-log", highlight=True)
                )

            # 5. Rhizome
            with TabPane("Rhizome", id="tab-rhizome"):
                yield Horizontal(
                    Vertical(
                        Input(placeholder="Search (Enter to query)...", id="rhizome-search"),
                        DataTable(id="rhizome-artifacts"),
                        Horizontal(
                            Button("Promote to Git", id="btn-tui-promote", variant="primary"),
                            Button("Toggle Hex", id="btn-toggle-hex", variant="default"),
                            classes="button-row"
                        ),
                        classes="rhizome-left-pane"
                    ),
                    ScrollableContainer(
                        Markdown("Select an artifact to view content", id="rhizome-content-view"),
                        classes="rhizome-right-pane"
                    )
                )

            # 4. Services
            with TabPane("Services", id="tab-svc"):
                yield DataTable(id="svc-table")
                yield Horizontal(
                    Button("Start", id="btn-start", variant="success"),
                    Button("Stop", id="btn-stop", variant="error"),
                    Button("Restart", id="btn-restart", variant="warning"),
                    classes="button-row"
                )

            # 5. SemOps
            with TabPane("SemOps", id="tab-semops"):
                yield Vertical(
                    DataTable(id="semops-table"),
                    Horizontal(
                        Input(placeholder="KEY (e.g. MYOP)", id="semops-key"),
                        Input(placeholder="id (e.g. myop)", id="semops-id"),
                        Input(placeholder="domain", id="semops-domain"),
                        Input(placeholder="category", id="semops-category"),
                        classes="button-row"
                    ),
                    Horizontal(
                        Input(placeholder="aliases (comma-separated)", id="semops-aliases"),
                        Input(placeholder="tool (path)", id="semops-tool"),
                        Input(placeholder="bus_topic", id="semops-bus-topic"),
                        classes="button-row"
                    ),
                    Horizontal(
                        Input(placeholder="ui.route", id="semops-ui-route"),
                        Input(placeholder="ui.component", id="semops-ui-component"),
                        Input(placeholder="agents (comma)", id="semops-agents"),
                        Input(placeholder="apps (comma)", id="semops-apps"),
                        classes="button-row"
                    ),
                    Horizontal(
                        Input(placeholder="description", id="semops-description"),
                        Button("Save", id="btn-semops-save", variant="success"),
                        Button("Remove", id="btn-semops-remove", variant="error"),
                        classes="button-row"
                    ),
                )

            # 6. Policy
            with TabPane("Policy", id="tab-policy"):
                yield ScrollableContainer(
                    Markdown("Loading policy artifacts...", id="policy-content"),
                    classes="policy-pane"
                )

            # 7. PBDEEP
            with TabPane("PBDEEP", id="tab-pbdeep"):
                yield Vertical(
                    Markdown("Awaiting PBDEEP report...", id="pbdeep-summary"),
                    Log(id="pbdeep-log", highlight=True),
                    classes="pbdeep-pane"
                )

            # 8. Events
            with TabPane("Events", id="tab-events"):
                yield Log(id="bus-log", highlight=True)

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#svc-table", DataTable).add_columns("ID", "Name", "Status", "PID")
        self.query_one("#rhizome-artifacts", DataTable).add_columns("SHA", "Name", "Kind", "Created")
        try:
            self.query_one("#git-log-table", DataTable).add_columns("SHA", "Message", "Author", "Date")
        except Exception:
            pass
        self.query_one("#semops-table", DataTable).add_columns("KEY", "id", "domain", "category", "aliases", "tool", "bus_topic", "user")
        
        self.set_interval(1.0, self.poll_bus)
        self.set_interval(5.0, self.refresh_services)
        self.set_interval(10.0, self.load_rhizome_artifacts)
        self.set_interval(10.0, self.load_git_status)
        self.set_interval(15.0, self.load_semops_table)
        
        # Plurichat specific updates
        self.set_interval(2.0, self.update_plurichat_status)
        self.set_interval(5.0, self.update_plurichat_agent_activity)

        # Initial logs
        self.log_bus_event("System initialized.", "INFO")
        self.load_semops_table()
        self.load_git_status()
        self.load_policy_doc()
        self.update_plurichat_status() # Initial call
        if self.motd_mode:
            try:
                self.push_screen(
                    MotdScreen(
                        root=self.root,
                        bus_dir=getattr(self, "bus_dir", None),
                        state_root=getattr(self, "state_root", None),
                        semops_path=getattr(self, "SEMOPS_PATH", None),
                        registry=self.registry,
                    )
                )
            except Exception:
                pass


    def load_semops_table(self) -> None:
        """Load semantic operators (built-in + user-defined) into SemOps table."""
        if not SemopsLexer: return
        try:
            lexer = SemopsLexer(semops_path=self.SEMOPS_PATH, user_ops_path=self.USER_OPS_PATH)
            table = self.query_one("#semops-table", DataTable)
            table.clear()
            for key, op in sorted(lexer.operators.items(), key=lambda kv: kv[0].lower()):
                table.add_row(
                    key,
                    op.id,
                    op.domain,
                    op.category,
                    ", ".join(op.aliases or []),
                    op.tool or "",
                    op.bus_topic or "",
                    "user" if op.user_defined else "builtin",
                    key=key,
                )
        except Exception as e:
            try:
                self.query_one("#dialogos-log", Log).write(f"[red]SemOps load failed: {e}[/]")
            except Exception:
                pass

    def load_git_status(self) -> None:
        """Populate the git log table if it exists."""
        try:
            self.query_one("#git-log-table", DataTable)
        except Exception:
            return

    def update_plurichat_status(self) -> None:
        return

    def update_plurichat_agent_activity(self) -> None:
        return

    def load_policy_doc(self) -> None:
        """Load foundation spec + policy map into the Policy tab."""
        spec_path = (self.root / "nucleus" / "specs" / "pluribus_foundation.md").resolve()
        map_path = (self.root / "nucleus" / "docs" / "policy" / "policy_map.json").resolve()
        parts: list[str] = []
        if spec_path.exists():
            try:
                parts.append(spec_path.read_text(encoding="utf-8"))
            except Exception as exc:
                parts.append(f"Failed to read foundation spec: {exc}")
        else:
            parts.append("Foundation spec not found.")

        if map_path.exists():
            try:
                import json as _json
                raw = _json.loads(map_path.read_text(encoding="utf-8"))
                pretty = _json.dumps(raw, indent=2, ensure_ascii=True)
                parts.append("\n## Policy Map (snapshot)\n```json\n" + pretty + "\n```")
            except Exception as exc:
                parts.append(f"Failed to read policy map: {exc}")
        else:
            parts.append("Policy map not found.")

        try:
            self.query_one("#policy-content", Markdown).update("\n\n".join(parts))
        except Exception:
            pass

    def hexdump(self, content: str | bytes) -> str:
        """Generate a canonical hex dump."""
        if isinstance(content, str):
            b = content.encode("utf-8", errors="replace")
        else:
            b = content
        
        lines = []
        for i in range(0, len(b), 16):
            chunk = b[i:i+16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            text_part = "".join((chr(c) if 32 <= c < 127 else ".") for c in chunk)
            lines.append(f"{i:08x}  {hex_part:<48}  |{text_part}|")
        return "\n".join(lines)

    def load_rhizome_artifacts(self) -> None:
        """Load rhizome artifacts from ndjson OR rag_index query."""
        table = self.query_one("#rhizome-artifacts", DataTable)
        
        if self.search_query:
            # RAG Search Mode
            import subprocess
            try:
                rag_tool = Path(__file__).parent / "rag_index.py"
                cmd = [sys.executable, str(rag_tool), "query", self.search_query, "-k", "20"]
                res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                
                table.clear()
                for line in res.stdout.splitlines():
                    try:
                        hit = json.loads(line)
                        sha = hit.get("doc_id", "?") # rag_index returns doc_id (uuid) or sha depending on implementation. Assuming doc_id for now.
                        name = hit.get("title") or "Unknown"
                        kind = "hit"
                        score = f"{hit.get('distance', 0):.2f}"
                        table.add_row(sha[:8], name, kind, score, key=sha)
                    except:
                        pass
                return
            except Exception as e:
                self.notify(f"Search failed: {e}", severity="error")

        # Default Mode: File listing
        if not self.ARTIFACTS_NDJSON_PATH.exists():
            return
        
        table.clear()
        
        artifacts = []
        try:
            with self.ARTIFACTS_NDJSON_PATH.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        art = json.loads(line)
                        artifacts.append(art)
                    except:
                        pass
        except Exception:
            pass
        
        # Sort by creation date (newest first)
        artifacts.sort(key=lambda x: x.get("ts", 0), reverse=True)

        for art in artifacts:
            sha = art.get("sha256", "?")
            sources = art.get("sources", [])
            name = sources[0].split("/")[-1] if sources else art.get("id", "?")
            if not name or name == "?":
                 name = art.get("filename", sha[:8])
            kind = art.get("kind", "?")
            created = art.get("iso", "?")[:10]
            table.add_row(sha[:6], name, kind, created, key=sha)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Load artifact content into view."""
        sha = str(event.row_key.value)
        
        # Path structure: .pluribus/objects/ab/cd/abcdef...
        if len(sha) >= 4:
            artifact_content_path = self.RHIZOME_CONTENT_PATH / sha[:2] / sha[2:4] / sha
        else:
            artifact_content_path = self.RHIZOME_CONTENT_PATH / sha

        view = self.query_one("#rhizome-content-view", Markdown)
        if not artifact_content_path.exists():
             # Try flat structure as fallback
             artifact_content_path = self.RHIZOME_CONTENT_PATH / sha

        if not artifact_content_path.exists():
            view.update(f"Preview not available for {sha}.\n(Object not in cache)")
            return

        try:
            # Read bytes for hex mode support
            raw_content = artifact_content_path.read_bytes()
            
            if self.view_mode == "hex":
                view.update(f"```text\n{self.hexdump(raw_content)}\n```")
            else:
                try:
                    text_content = raw_content.decode("utf-8")
                    view.update(f"```\n{text_content}\n```")
                except UnicodeDecodeError:
                    view.update(f"```text\n{self.hexdump(raw_content)}\n```")
        except Exception as e:
            view.update(f"Error reading artifact content: {e}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-tui-promote":
            self.action_promote()
        elif event.button.id == "btn-toggle-hex":
            self.view_mode = "hex" if self.view_mode == "markdown" else "markdown"
            self.notify(f"View Mode: {self.view_mode}", severity="information")
            # Trigger refresh of current view if possible, or wait for next selection
            table = self.query_one("#rhizome-artifacts", DataTable)
            try:
                if table.cursor_row_key:
                    # Re-trigger selection logic to refresh view
                    self.on_data_table_row_selected(DataTable.RowSelected(table, table.cursor_row_key))
            except:
                pass
        
        elif event.button.id == "btn-force-muse":
            # Emit high entropy event to trigger mood shift
            emit_event(
                self.bus_paths,
                topic="mabswarm.probe", # Director listens to this
                kind="log",
                level="error", # Fake error to spike anxiety -> mood change
                actor="tui-user",
                data={"probe": "manual_art_trigger", "entropy_injection": 999},
                trace_id=None,
                run_id=None,
                durable=False
            )
            self.notify("Muse Provoked (injected entropy)", severity="warning")

        elif event.button.id == "btn-voice-start":
            emit_event(
                self.bus_paths,
                topic="voice.stt.start",
                kind="metric",
                level="info",
                actor="tui-user",
                data={"req_id": f"voice-stt-{uuid.uuid4()}", "iso": datetime.utcnow().isoformat()},
                trace_id=None,
                run_id=None,
                durable=False,
            )
            try:
                self.query_one("#voice-log", Log).write("[green]STT start requested[/]")
            except:
                pass

        elif event.button.id == "btn-voice-stop":
            emit_event(
                self.bus_paths,
                topic="voice.stt.stop",
                kind="metric",
                level="info",
                actor="tui-user",
                data={"req_id": f"voice-stt-{uuid.uuid4()}", "iso": datetime.utcnow().isoformat()},
                trace_id=None,
                run_id=None,
                durable=False,
            )
            try:
                self.query_one("#voice-log", Log).write("[yellow]STT stop requested[/]")
            except:
                pass

        elif event.button.id == "btn-voice-tts":
            try:
                import os
                import subprocess
                import json as _json
                speak_tool = (self.root / "nucleus" / "tools" / "speak_operator.py").resolve()
                if not speak_tool.exists():
                    raise FileNotFoundError(str(speak_tool))
                payload = {
                    "origin": "tui",
                    "intent": "voice_tts_test",
                }
                subprocess.run(
                    [
                        sys.executable,
                        str(speak_tool),
                        "--file",
                        str(self.root / "speaker_bus.aiff"),
                        "--emit-bus",
                        "--broadcast",
                        "--context-json",
                        _json.dumps(payload, ensure_ascii=True),
                        "--source",
                        "tui",
                        "--reason",
                        "tui_voice_tts",
                    ],
                    input="TUI voice test. Speaker bus event queued.",
                    text=True,
                    check=False,
                    timeout=5.0,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.notify("TTS test queued (SPEAK)", severity="information")
            except Exception as exc:
                self.notify(f"TTS test failed: {str(exc)[:160]}", severity="error")

        elif event.button.id in {"btn-semops-save", "btn-semops-remove"}:
             self.handle_semops_button(event)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command interjection and plurichat input."""
        if event.input.id == "cmd-input":
            cmd = event.value
            if not cmd: return
            
            emit_event(
                self.bus_paths,
                topic="strp.request",
                kind="request",
                level="info",
                actor="tui-user",
                data={
                    "goal": cmd,
                    "provider_hint": "auto",
                    "kind": "chat",
                    "origin": "tui"
                },
                trace_id=None,
                run_id=None,
                durable=True
            )
            
            self.query_one("#dialogos-log", Log).write(f"[bold green]USER: {cmd}[/]")
            event.input.value = ""
            self.log_bus_event(f"Interjected command: {cmd}", "ACTION")
            
        elif event.input.id == "rhizome-search":
            self.search_query = event.value
            self.load_rhizome_artifacts()
            self.notify(f"Searching: {self.search_query}", severity="information")

        elif event.input.id == "plurichat-input":
            prompt = event.value.strip()
            if not prompt: return
            
            transcript_log = self.query_one("#plurichat-transcript", Log)
            transcript_log.write(f"[bold green]YOU:[/]{prompt}")
            event.input.value = "" # Clear input immediately
            self.chat_input_buf = "" # Clear reactive too
            
            # Execute chat in a separate thread to avoid blocking the TUI
            def chat_thread():
                try:
                    # Use existing chat_state for routing and configuration
                    response = execute_with_topology(
                        prompt,
                        select_provider_for_query(prompt, self.chat_state.provider, self.chat_state.providers),
                        self.bus_paths.bus_dir,
                        self.chat_state.actor,
                        self.chat_state.mode
                    )
                    
                    if response.success:
                        transcript_log.write(f"[blue]ASSISTANT:[/]{response.text}")
                    else:
                        transcript_log.write(f"[red]ERROR:[/]{response.error or 'Unknown chat error'}")
                        if response.text:
                            transcript_log.write(f"[red]DETAILS:[/]{response.text}")
                except Exception as e:
                    transcript_log.write(f"[red]CRITICAL CHAT ERROR:[/]{e}")
            
            import threading
            threading.Thread(target=chat_thread).start()

    def action_promote(self) -> None:
        """Trigger promotion flow."""
        table = self.query_one("#rhizome-artifacts", DataTable)
        try:
            selected_row_key = table.cursor_row_key
        except Exception:
            selected_row_key = None
            
        if not selected_row_key: 
            self.notify("No artifact selected for promotion.", severity="warning")
            return

        artifact_sha = str(selected_row_key.value)
        
        req_id = f"req-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(4).hex()}"
        self.active_plan_req_id = req_id # Start tracking
        
        self.notify(f"Requesting plan for {artifact_sha[:6]} ({req_id})...", severity="information")
        
        emit_event(
            self.bus_paths,
            topic="repo.plan.request",
            kind="request",
            level="info",
            actor="tui-user",
            data={
                "req_id": req_id,
                "artifact_sha": artifact_sha,
                "template_id": "default"
            },
            trace_id=None,
            run_id=None,
            durable=True
        )

    def handle_semops_button(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if not SemopsLexer: return

        key = (self.query_one("#semops-key", Input).value or "").strip().upper()
        if not key:
            self.notify("SemOps KEY required", severity="warning")
            return

        lexer = SemopsLexer(semops_path=self.SEMOPS_PATH, user_ops_path=self.USER_OPS_PATH)
        if bid == "btn-semops-remove":
            ok = lexer.undefine_operator(key)
            if ok:
                self.notify(f"Removed {key}", severity="information")
                self.load_semops_table()
            else:
                self.notify(f"Cannot remove {key}", severity="warning")
            return

        # Save/update logic (simplified for brevity)
        self.notify("SemOps Save not fully implemented in this TUI revision", severity="warning")

    def format_dimensional_event(self, evt: dict) -> str:
        """Format event with dimensional richness for TUI display."""
        topic = evt.get("topic", "")
        kind = evt.get("kind", "log")
        level = evt.get("level", "info")
        actor = evt.get("actor", "")
        iso = evt.get("iso", evt.get("temporal", {}).get("iso", ""))[:19] if evt.get("iso") or evt.get("temporal") else ""

        # Level colors
        level_colors = {"debug": "dim", "info": "cyan", "warn": "yellow", "error": "red bold"}
        level_style = level_colors.get(level, "white")
        
        base = f"[{level_style}]{iso}[/] [{level_style}]{topic}[/] ({kind})"
        if actor:
            base += f" [dim]@{actor}[/]"
        return base

    def format_voice_event(self, evt: dict) -> str:
        """Format SPEAK events for the Voice log."""
        data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
        broadcast = data.get("broadcast") if isinstance(data.get("broadcast"), dict) else {}
        text = data.get("text_excerpt") or data.get("text") or ""
        if isinstance(text, str) and len(text) > 180:
            text = text[:180] + "..."
        iso = (evt.get("iso") or "")[:19]
        actor = evt.get("actor") or "unknown"
        status = broadcast.get("status") if isinstance(broadcast.get("status"), str) else "queued"
        return f"[cyan]{iso}[/] [green]SPEAK[/] [dim]@{actor}[/] [{status}] {text}"

    def format_pbdeep_event(self, evt: dict) -> str:
        """Format PBDEEP events for the PBDEEP log."""
        data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
        iso = (evt.get("iso") or "")[:19]
        req_id = str(data.get("req_id") or "")[:8]
        topic = evt.get("topic", "")
        if topic == "operator.pbdeep.progress":
            stage = data.get("stage") or "stage"
            status = data.get("status") or "status"
            percent = data.get("percent") or 0
            return f"[cyan]{iso}[/] [green]PBDEEP[/] [dim]{req_id}[/] {stage} {status} {percent}%"
        if topic == "operator.pbdeep.index.updated":
            summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
            status = summary.get("status") or "index"
            return f"[cyan]{iso}[/] [green]PBDEEP[/] [dim]{req_id}[/] index {status}"
        if topic == "operator.pbdeep.report":
            mode = data.get("mode") or "report"
            return f"[cyan]{iso}[/] [green]PBDEEP[/] [dim]{req_id}[/] report mode={mode}"
        return f"[cyan]{iso}[/] [green]PBDEEP[/] [dim]{req_id}[/] request"

    def render_pbdeep_summary(self, evt: dict) -> str:
        """Render PBDEEP summary markdown."""
        data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        req_id = data.get("req_id") or ""
        mode = data.get("mode") or ""
        iso = (data.get("iso") or evt.get("iso") or "")[:19]
        lines = [
            "**PBDEEP Report**",
            "",
            f"- req_id: `{req_id}`",
            f"- iso: `{iso}`",
            f"- mode: `{mode}`",
            "",
            "## Summary",
            f"- branches_total: {summary.get('branches_total', 0)}",
            f"- final_branches: {summary.get('final_branches', 0)}",
            f"- lost_and_found_count: {summary.get('lost_and_found_count', 0)}",
            f"- untracked_count: {summary.get('untracked_count', 0)}",
            f"- doc_missing_count: {summary.get('doc_missing_count', 0)}",
            f"- code_missing_docs: {summary.get('code_missing_docs', 0)}",
        ]
        next_actions = summary.get("next_actions") or []
        lines.append("")
        lines.append("## Next Actions")
        if isinstance(next_actions, list) and next_actions:
            for action in next_actions:
                lines.append(f"- {action}")
        else:
            lines.append("- Awaiting next actions.")
        return "\n".join(lines)

    def poll_bus(self) -> None:
        if not self.bus_paths or not Path(self.bus_paths.events_path).exists():
            return

        try:
            import json
            with open(self.bus_paths.events_path, "r", errors="ignore") as f:
                lines = f.readlines()
                self.bus_events = len(lines)

                # Show last few in log
                bus_log = self.query_one("#bus-log", Log)
                bus_log.clear() 
                
                # Check for relevant events in the tail (last 50 for efficiency)
                tail = lines[-50:]
                voice_events: list[dict] = []
                pbdeep_events: list[dict] = []
                pbdeep_report: dict | None = None
                for line in tail:
                    try:
                        evt = json.loads(line)
                        topic = evt.get("topic", "")
                        data = evt.get("data", {})
                        
                        # Check for Plan Response
                        if self.active_plan_req_id and topic == "repo.plan.response":
                            # Check if it matches our req_id
                            # The event structure might have req_id in data or as a top-level field depending on emitter
                            e_req_id = data.get("req_id") or data.get("reqId")
                            if e_req_id == self.active_plan_req_id:
                                # Found our plan! Show modal.
                                self.active_plan_req_id = None # Stop tracking
                                self.push_screen(PlanPreviewModal(data, e_req_id, self.bus_paths))
                        
                        # Check for Exec Result
                        if topic == "repo.exec.result":
                             # If we had a modal open, it would have handled the request. 
                             # Here we just notify completion.
                             status = data.get("status")
                             if status == "success":
                                 self.notify(f"Promotion Successful: {data.get('target_path')}", severity="information")
                             elif status == "failed":
                                 self.notify(f"Promotion Failed: {data.get('errors')}", severity="error")

                        # Update metrics
                        if topic == "agent.topology.chosen":
                            topo = data.get("topology", "single")
                            fanout = int(data.get("fanout", 1))
                            reason = data.get("reason", "unknown")
                            try:
                                tw = self.query_one(TopologyWidget)
                                tw.topology = topo
                                tw.fanout = fanout
                                tw.reason = reason
                            except Exception:
                                pass

                        if topic == "pluribus.check.report":
                            self.vor_cdi = float(data.get("vor_metrics", {}).get("cdi", 0.0))
                        
                        if topic == "art.state.update":
                            self.current_mood = data.get("mood", "calm")
                            self.entropy = float(data.get("entropy", 0.0))
                            self.anxiety = float(data.get("anxiety", 0.0))
                            try:
                                self.query_one("#muse-status", Label).update(f"Ω Muse: {self.current_mood.upper()}")
                                self.query_one(DynamicHeader).update_art(self.current_mood, self.entropy, self.current_composition)
                            except:
                                pass
                        
                        if topic == "art.composition.change":
                            self.current_composition = data.get("composition", {})
                            try:
                                self.query_one(DynamicHeader).update_art(self.current_mood, self.entropy, self.current_composition)
                            except:
                                pass

                        if topic == "speaker.bus.write":
                            voice_events.append(evt)
                        if topic.startswith("operator.pbdeep."):
                            pbdeep_events.append(evt)
                            if topic == "operator.pbdeep.report":
                                pbdeep_report = evt

                        bus_log.write(self.format_dimensional_event(evt))
                    except:
                        pass

                try:
                    voice_log = self.query_one("#voice-log", Log)
                    voice_log.clear()
                    for evt in voice_events[-12:]:
                        voice_log.write(self.format_voice_event(evt))
                    if voice_events:
                        latest = voice_events[-1]
                        iso = (latest.get("iso") or "")[:19]
                        self.query_one("#voice-status", Markdown).update(
                            f"STT/TTS Pipeline Status: [green]READY[/green]\n\n**Latency:** 120ms\n**VAD:** Idle\n**Last SPEAK:** {iso}"
                        )
                except:
                    pass

                try:
                    pbdeep_log = self.query_one("#pbdeep-log", Log)
                    pbdeep_log.clear()
                    for evt in pbdeep_events[-16:]:
                        pbdeep_log.write(self.format_pbdeep_event(evt))
                    if pbdeep_report:
                        self.query_one("#pbdeep-summary", Markdown).update(self.render_pbdeep_summary(pbdeep_report))
                except:
                    pass
        except Exception:
            pass

    def refresh_services(self) -> None:
        if not self.registry: return
        self.registry.load()
        table = self.query_one("#svc-table", DataTable)
        table.clear()
        
        svc_map = {s.id: {"def": s, "inst": None} for s in self.registry.list_services()}
        for inst in self.registry.list_instances():
            if inst.service_id in svc_map:
                svc_map[inst.service_id]["inst"] = inst
        
        for sid, info in svc_map.items():
            s = info["def"]
            i = info["inst"]
            status = "stopped"
            pid = "-"
            if i and i.status == "running":
                status = "[green]running[/green]"
                pid = str(i.pid)
            elif i and i.status == "error":
                status = "[red]error[/red]"
            
            table.add_row(sid, s.name, status, pid, key=sid)

    def log_bus_event(self, msg: str, level: str) -> None:
        pass

    def action_escape(self) -> None:
        if isinstance(self.screen, MotdScreen):
            self.pop_screen()
            return
        if self._esc_armed:
            self.exit()
            return
        self._esc_armed = True
        try:
            self.notify("Press ESC again to exit", severity="warning", timeout=1.5)
        except Exception:
            pass
        self.set_timer(1.5, self._reset_escape)

    def _reset_escape(self) -> None:
        self._esc_armed = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("PLURIBUS_ROOT") or os.getcwd())
    parser.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR") or None)
    parser.add_argument("--motd", action="store_true")
    args = parser.parse_args()
    
    root = Path(args.root).resolve()
    # Find true root if in subdir
    for p in [root, *root.parents]:
        if (p / ".pluribus").exists():
            root = p
            break
            
    bus_dir_default = args.bus_dir or str(root / ".pluribus" / "bus")
    
    # Allow running without valid bus for testing
    if resolve_bus_paths:
        try:
            bus_dir = Path(resolve_bus_paths(bus_dir_default).active_dir)
            app = PluribusTUI(root, bus_dir, motd_mode=args.motd)
            app.run()
            return
        except Exception as e:
            print(f"Bus resolution failed: {e}")
    
    # Fallback/Test mode
    app = PluribusTUI(root, None, motd_mode=args.motd)
    app.run()

if __name__ == "__main__":
    main()
