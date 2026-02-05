#!/usr/bin/env python3
"""
bus_monitor_tui.py - Elegant Bus Monitor
Displays bus events in a clean table.
"""
import sys
import json
import time
import os
from collections import deque

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.syntax import Syntax
except ImportError:
    print("Rich not installed. Please install rich: pip install rich")
    sys.exit(1)

BUS_FILE = ".pluribus/bus/events.ndjson"
MAX_ROWS = 15

def tail_bus():
    if not os.path.exists(BUS_FILE):
        return []
    
    events = []
    # Simple tail of last N lines
    try:
        with open(BUS_FILE, "r") as f:
            lines = f.readlines()
            for line in lines[-MAX_ROWS:]:
                try:
                    events.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass  # Skip malformed JSON lines
    except Exception:
        pass
    return list(reversed(events))

def make_table(events):
    table = Table(expand=True, header_style="bold magenta", border_style="dim")
    table.add_column("Time", style="cyan", width=10)
    table.add_column("Actor", style="green", width=12)
    table.add_column("Topic", style="yellow", width=25)
    table.add_column("Kind", style="blue", width=8)
    table.add_column("Data Preview", style="white", no_wrap=True)

    for evt in events:
        ts = evt.get("iso", "").split("T")[-1][:8]
        actor = evt.get("actor", "unknown")[:12]
        topic = evt.get("topic", "")[:25]
        kind = evt.get("kind", "")[:8]
        data = json.dumps(evt.get("data", {}))
        
        # Truncate data cleanly
        if len(data) > 60:
            data = data[:57] + "..."
            
        table.add_row(ts, actor, topic, kind, data)
        
    return table

def main():
    console = Console()
    layout = Layout()
    
    with Live(console=console, screen=True, refresh_per_second=4) as live:
        while True:
            events = tail_bus()
            table = make_table(events)
            
            # If we had interaction (mouse/keyboard), we'd update a detail panel here.
            # For now, just the clean table.
            
            live.update(Panel(table, title="[bold]Pluribus Bus Monitor[/bold] (Live)", border_style="blue"))
            time.sleep(0.25)

if __name__ == "__main__":
    main()
