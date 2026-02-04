#!/usr/bin/env python3
"""Services Management TUI.

Interactive dashboard for managing Pluribus services via service_registry.py.
Launch, stop, and monitor services from a terminal UI.

Usage:
    python3 nucleus/tools/services_tui.py [--root /path/to/.pluribus]

Keybindings:
    q - Quit
    s - Start selected service
    t - Stop selected service
    h - Health check selected service
    r - Refresh
    enter - Toggle service selection
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
        Static,
    )
    from textual.binding import Binding
except ImportError:
    sys.stderr.write("ERROR: textual not installed. Run: pip install textual\n")
    sys.exit(1)

# Import service registry
sys.path.insert(0, str(Path(__file__).parent))
from service_registry import ServiceRegistry, ServiceDef, ServiceInstance, find_rhizome_root


def find_pluribus_root(start: Path) -> Path | None:
    """Find .pluribus directory by walking upward."""
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand / ".pluribus"
        if (cand / ".pluribus").is_dir():
            return cand / ".pluribus"
    return None


class ServicesTUI(App):
    """Services Management TUI."""

    # Force dark mode for better visibility
    dark = True

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1 3;
        grid-rows: 3fr 2fr auto;
    }

    #services-panel {
        border: solid cyan;
        height: 100%;
    }

    #instances-panel {
        border: solid green;
        height: 100%;
    }

    #control-panel {
        border: solid magenta;
        height: auto;
        padding: 0 1;
    }

    #button-row {
        height: 3;
        width: 100%;
    }

    #status-bar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }

    .panel-title {
        text-style: bold;
        background: $surface;
        padding: 0 1;
    }

    DataTable {
        height: 100%;
    }

    Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "start_service", "Start"),
        Binding("t", "stop_service", "Stop"),
        Binding("h", "health_check", "Health"),
    ]

    TITLE = "Pluribus Services"
    SUB_TITLE = "Service Management"

    def __init__(self, pluribus_root: Path):
        super().__init__()
        self.pluribus_root = pluribus_root
        self.project_root = pluribus_root.parent
        self.registry = ServiceRegistry(self.project_root)
        self.registry.init()
        self.registry.load()
        self.selected_service: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Services Panel
        yield Vertical(
            Static("Registered Services", classes="panel-title"),
            DataTable(id="services-table"),
            id="services-panel",
        )

        # Running Instances Panel
        yield Vertical(
            Static("Running Instances", classes="panel-title"),
            DataTable(id="instances-table"),
            id="instances-panel",
        )

        # Control Panel
        yield Vertical(
            Horizontal(
                Button("Start (s)", id="start-btn", variant="success"),
                Button("Stop (t)", id="stop-btn", variant="error"),
                Button("Health (h)", id="health-btn", variant="primary"),
                Button("Refresh (r)", id="refresh-btn", variant="default"),
                id="button-row",
            ),
            Static("", id="status-bar"),
            id="control-panel",
        )

        yield Footer()

    def on_mount(self) -> None:
        # Initialize services table
        services_table = self.query_one("#services-table", DataTable)
        services_table.cursor_type = "row"
        services_table.add_columns("ID", "Name", "Kind", "Port", "Tags", "Status")

        # Initialize instances table
        instances_table = self.query_one("#instances-table", DataTable)
        instances_table.add_columns("Service", "Instance ID", "PID", "Port", "Status", "Health", "Started")

        # Load initial data
        self.refresh_services()
        self.refresh_instances()

        # Set up polling
        self.set_interval(5.0, self.refresh_instances)

        self.update_status("Ready. Select a service and press 's' to start, 't' to stop.")

    def update_status(self, message: str) -> None:
        """Update the status bar message."""
        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(f"[dim]{message}[/dim]")

    def refresh_services(self) -> None:
        """Refresh services table."""
        self.registry.load()
        table = self.query_one("#services-table", DataTable)
        table.clear()

        for svc_id, svc in sorted(self.registry._services.items()):
            # Check if running
            running = False
            for inst in self.registry._instances.values():
                if inst.service_id == svc_id and inst.status == "running":
                    running = True
                    break

            status = "[green]running[/green]" if running else "[dim]stopped[/dim]"
            port_str = str(svc.port) if svc.port else "-"
            tags_str = ", ".join(svc.tags[:3])
            if len(svc.tags) > 3:
                tags_str += f" +{len(svc.tags)-3}"

            table.add_row(
                svc_id,
                svc.name,
                svc.kind,
                port_str,
                tags_str,
                status,
                key=svc_id,
            )

    def refresh_instances(self) -> None:
        """Refresh instances table."""
        self.registry.load()
        table = self.query_one("#instances-table", DataTable)
        table.clear()

        for inst_id, inst in sorted(self.registry._instances.items()):
            health_style = {
                "healthy": "[green]healthy[/green]",
                "unhealthy": "[red]unhealthy[/red]",
                "unknown": "[yellow]unknown[/yellow]",
            }.get(inst.health, inst.health)

            status_style = {
                "running": "[green]running[/green]",
                "stopped": "[dim]stopped[/dim]",
                "error": "[red]error[/red]",
                "starting": "[yellow]starting[/yellow]",
            }.get(inst.status, inst.status)

            table.add_row(
                inst.service_id,
                inst_id[:12],
                str(inst.pid or "-"),
                str(inst.port or "-"),
                status_style,
                health_style,
                inst.started_iso[:16] if inst.started_iso else "-",
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in services table."""
        if event.data_table.id == "services-table":
            if event.row_key:
                self.selected_service = str(event.row_key.value)
                svc = self.registry._services.get(self.selected_service)
                if svc:
                    self.update_status(f"Selected: {svc.name} ({svc.kind}). Press 's' to start, 't' to stop.")

    def action_refresh(self) -> None:
        """Refresh all data."""
        self.refresh_services()
        self.refresh_instances()
        self.update_status("Refreshed.")

    def action_start_service(self) -> None:
        """Start selected service."""
        if not self.selected_service:
            self.notify("No service selected", severity="warning")
            return

        svc = self.registry._services.get(self.selected_service)
        if not svc:
            self.notify(f"Service not found: {self.selected_service}", severity="error")
            return

        try:
            instance = self.registry.start_service(self.selected_service)
            self.refresh_services()
            self.refresh_instances()
            if instance and instance.status == "running":
                self.notify(f"Started {svc.name}: {instance.instance_id}", severity="information")
                self.update_status(f"Started {svc.name}")
            elif instance:
                self.notify(f"Start failed: {instance.error}", severity="error")
            else:
                self.notify("Failed to start service", severity="error")
        except Exception as e:
            self.notify(f"Failed to start: {e}", severity="error")

    def action_stop_service(self) -> None:
        """Stop selected service."""
        if not self.selected_service:
            self.notify("No service selected", severity="warning")
            return

        svc = self.registry._services.get(self.selected_service)
        if not svc:
            self.notify(f"Service not found: {self.selected_service}", severity="error")
            return

        # Find running instance
        instance_id = None
        for inst_id, inst in self.registry._instances.items():
            if inst.service_id == self.selected_service and inst.status == "running":
                instance_id = inst_id
                break

        if not instance_id:
            self.notify(f"No running instance of {svc.name}", severity="warning")
            return

        try:
            self.registry.stop_service(instance_id)
            self.refresh_services()
            self.refresh_instances()
            self.notify(f"Stopped {svc.name}", severity="information")
            self.update_status(f"Stopped {svc.name}")
        except Exception as e:
            self.notify(f"Failed to stop: {e}", severity="error")

    def action_health_check(self) -> None:
        """Health check selected service."""
        if not self.selected_service:
            self.notify("No service selected", severity="warning")
            return

        svc = self.registry._services.get(self.selected_service)
        if not svc:
            self.notify(f"Service not found: {self.selected_service}", severity="error")
            return

        # Find running instance
        for inst_id, inst in self.registry._instances.items():
            if inst.service_id == self.selected_service and inst.status == "running":
                try:
                    health = self.registry.check_health(inst_id)
                    self.refresh_instances()
                    is_healthy = health == "healthy"
                    self.notify(f"{svc.name}: {health}", severity="information" if is_healthy else "warning")
                except Exception as e:
                    self.notify(f"Health check failed: {e}", severity="error")
                return

        self.notify(f"No running instance of {svc.name}", severity="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        if button_id == "start-btn":
            self.action_start_service()
        elif button_id == "stop-btn":
            self.action_stop_service()
        elif button_id == "health-btn":
            self.action_health_check()
        elif button_id == "refresh-btn":
            self.action_refresh()


def main() -> int:
    parser = argparse.ArgumentParser(description="Services Management TUI")
    parser.add_argument("--root", help="Path to .pluribus directory")
    args = parser.parse_args()

    if args.root:
        pluribus_root = Path(args.root).expanduser().resolve()
    else:
        pluribus_root = find_pluribus_root(Path.cwd())
        if not pluribus_root:
            sys.stderr.write("ERROR: Could not find .pluribus directory\n")
            sys.exit(1)

    app = ServicesTUI(pluribus_root)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
