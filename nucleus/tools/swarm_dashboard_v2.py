#!/usr/bin/env python3
"""
swarm_dashboard_v2.py - Portal Inception Dashboard

Visualizes the Portal Inception Infrastructure:
- Portal Inception Status (PBPORTAL)
- Rhizome DAG Topology
- OHM Health & Metrics
- HEXIS Buffer Activity
- Etymon Registry

Ring: 1 (Infrastructure)
Protocol: DKIN v29 | PAIP v15 | Citizen v1
"""

import curses
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration
OHM_PATH = Path("nucleus/tools/ohm.py")
RHIZOME_PATH = Path("nucleus/tools/rhizome.py")
PORTAL_LEDGER = Path(".pluribus/portals/portals.ndjson")
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))


class SwarmDashboard:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        curses.start_color()
        curses.use_default_colors()
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        
        # Colors
        curses.init_pair(1, curses.COLOR_GREEN, -1)   # Success
        curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Warning/Pending
        curses.init_pair(3, curses.COLOR_RED, -1)     # Error
        curses.init_pair(4, curses.COLOR_CYAN, -1)    # Info/Header
        curses.init_pair(5, curses.COLOR_MAGENTA, -1) # Portal/Magic

        self.last_refresh = 0
        self.portals = []
        self.bus_events = []
        
    def load_data(self):
        """Load data from ledgers and bus."""
        # Load portals
        self.portals = []
        if PORTAL_LEDGER.exists():
            try:
                raw = PORTAL_LEDGER.read_text().strip().split("\n")
                self.portals = [json.loads(line) for line in raw if line][-10:]
            except Exception:
                pass
                
        # Load recent bus events
        event_path = BUS_DIR / "events.ndjson"
        self.bus_events = []
        if event_path.exists():
            try:
                # Read last 50 lines efficiently
                lines = subprocess.check_output(
                    ["tail", "-50", str(event_path)]
                ).decode().strip().split("\n")
                for line in lines:
                    if line:
                        evt = json.loads(line)
                        if evt.get("topic") in [
                            "portal.incepted", "rhizome.node.created",
                            "a2a.handshake.ack", "ohm.guardian.warn"
                        ]:
                            self.bus_events.append(evt)
            except Exception:
                pass
        self.bus_events = self.bus_events[-15:]

    def draw_header(self):
        h, w = self.stdscr.getmaxyx()
        title = " PLURIBUS SWARM DASHBOARD v2 "
        self.stdscr.addstr(0, (w - len(title)) // 2, title, curses.color_pair(4) | curses.A_BOLD)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stdscr.addstr(0, w - len(timestamp) - 2, timestamp)
        
        self.stdscr.addstr(1, 2, f"Files: {len(self.portals)} Portals | Mode: WORKTREEPIVOT", curses.A_DIM)
        self.stdscr.hline(2, 0, curses.ACS_HLINE, w)

    def draw_portals_panel(self, y, x, h, w):
        self.stdscr.addstr(y, x + 2, " PORTAL INCEPTION ", curses.color_pair(5) | curses.A_BOLD)
        self.stdscr.box(y + 1, x, 0, 0) # Should use separate win, but this is simple demo
        
        for i, portal in enumerate(self.portals[:h-4]):
            try:
                pid = portal.get("portal_id", "")[:8]
                etymon = portal.get("etymon", "unknown")[:20]
                status = portal.get("status", "unknown")
                score = portal.get("cmp_score", 0.0)
                
                line = f"{pid} | {etymon:<20} | CMP:{score:.2f} | {status}"
                color = curses.color_pair(1) if status == "incepted" else curses.color_pair(2)
                self.stdscr.addstr(y + 2 + i, x + 2, line[:w-4], color)
            except Exception:
                pass

    def draw_bus_monitor(self, y, x, h, w):
        self.stdscr.addstr(y, x + 2, " BUS ACTIVITY ", curses.color_pair(4) | curses.A_BOLD)
        
        for i, evt in enumerate(reversed(self.bus_events)):
            if i >= h - 2: break
            try:
                topic = evt.get("topic", "")
                actor = evt.get("actor", "")[:10]
                ts = datetime.fromtimestamp(evt.get("ts", 0)).strftime("%H:%M:%S")
                
                color = curses.color_pair(2)
                if "ack" in topic: color = curses.color_pair(1)
                if "warn" in topic or "error" in topic: color = curses.color_pair(3)
                if "portal" in topic: color = curses.color_pair(5)
                
                line = f"{ts} [{actor:<10}] {topic}"
                self.stdscr.addstr(y + 2 + i, x + 2, line[:w-4], color)
            except Exception:
                pass

    def run(self):
        while True:
            self.stdscr.clear()
            h, w = self.stdscr.getmaxyx()
            
            self.load_data()
            self.draw_header()
            
            # Split screen vertical
            mid_x = w // 2
            
            # Left: Portals
            self.draw_portals_panel(3, 0, h - 4, mid_x)
            
            # Right: Bus
            self.draw_bus_monitor(3, mid_x, h - 4, w - mid_x)
            
            self.stdscr.refresh()
            
            # Input handling
            k = self.stdscr.getch()
            if k == ord('q'):
                break
            
            time.sleep(1)

def main():
    import subprocess
    curses.wrapper(lambda stdscr: SwarmDashboard(stdscr).run())

if __name__ == "__main__":
    main()
