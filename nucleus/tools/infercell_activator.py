#!/usr/bin/env python3
"""
InferCell Activator (Context Supervisor)
========================================

This daemon "activates" the static InferCell structures managed by `infercell_manager.py`.
It listens to the bus for cell lifecycle events and performs the physical/operational
setup required for an agent to actually work in that cell.

Responsibilities:
1.  **Cell Realization**: When a cell is created/forked, ensure its physical workspace exists.
    -   `.pluribus/infercells/<cell_id>/workspace/`
    -   `.pluribus/infercells/<cell_id>/memory.json`
2.  **Context Injection**: Pre-populate the workspace with artifacts from the parent cell (if forked).
3.  **Liveness Monitoring**: Watch active cells for activity (heartbeats).
4.  **Garbage Collection**: Archive/compress cells that are `merged` or `complete`.

Usage:
    python3 infercell_activator.py --bus-dir ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

# Import Manager for logic reuse (assuming it's in the same dir or path)
try:
    from infercell_manager import InferCellManager, InferCell
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from infercell_manager import InferCellManager, InferCell


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class InferCellActivator:
    def __init__(self, root: Path, bus_dir: Path, poll_interval: float = 1.0):
        self.root = root
        self.bus_dir = bus_dir
        self.poll_interval = poll_interval
        self.manager = InferCellManager(root)
        self.events_path = bus_dir / "events.ndjson"
        
        # Ensure we can track where we are in the event stream
        self.cursor_path = self.manager.storage_dir / "activator_cursor.txt"
        self.last_pos = 0
        if self.cursor_path.exists():
            try:
                self.last_pos = int(self.cursor_path.read_text().strip())
            except ValueError:
                self.last_pos = 0

    def _save_cursor(self):
        self.cursor_path.write_text(str(self.last_pos))

    def _emit(self, topic: str, kind: str, data: dict):
        tool = self.root / "nucleus" / "tools" / "agent_bus.py"
        subprocess.run(
            [
                sys.executable,
                str(tool),
                "--bus-dir",
                str(self.bus_dir),
                "pub",
                "--topic",
                topic,
                "--kind",
                kind,
                "--level",
                "info",
                "--actor",
                "infercell-activator",
                "--data",
                json.dumps(data, ensure_ascii=False),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _realize_cell(self, trace_id: str):
        """Physical setup for a cell."""
        cell = self.manager.get_cell(trace_id)
        if not cell:
            return

        cell_dir = self.manager.storage_dir / cell.cell_id
        workspace = cell_dir / "workspace"
        
        if not workspace.exists():
            workspace.mkdir(parents=True, exist_ok=True)
            self._emit("infercell.workspace.created", "metric", {
                "trace_id": trace_id,
                "cell_id": cell.cell_id,
                "path": str(workspace)
            })

            # Handle Fork Copying
            if cell.parent_trace_id:
                parent_cell = self.manager.get_cell(cell.parent_trace_id)
                if parent_cell:
                    parent_dir = self.manager.storage_dir / parent_cell.cell_id / "workspace"
                    if parent_dir.exists():
                        # Copy contents (Naively for now - could be optimized with COW/Hardlinks)
                        # In production, we'd use 'cp -al' for hardlinks if on same FS
                        try:
                            # Using shell for cp -r logic
                            subprocess.run(["cp", "-r", f"{parent_dir}/.", str(workspace)], check=True)
                            self._emit("infercell.workspace.cloned", "metric", {
                                "trace_id": trace_id,
                                "parent_trace_id": cell.parent_trace_id
                            })
                        except Exception as e:
                            self._emit("infercell.error", "error", {
                                "msg": "Failed to clone workspace",
                                "error": str(e)
                            })

    def run(self):
        print(f"InferCell Activator running (Bus: {self.bus_dir})")
        
        # Ensure events file exists
        if not self.events_path.exists():
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self.events_path.touch()

        while True:
            # Check for new events
            current_size = self.events_path.stat().st_size
            if current_size > self.last_pos:
                with open(self.events_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(self.last_pos)
                    for line in f:
                        self.last_pos += len(line.encode("utf-8")) # Approximate byte advance
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                            self._handle_event(evt)
                        except json.JSONDecodeError:
                            continue
                self._save_cursor()
            
            time.sleep(self.poll_interval)

    def _handle_event(self, evt: dict):
        topic = evt.get("topic", "")
        data = evt.get("data", {})
        
        if topic == "infercell.genesis":
            self._realize_cell(data.get("trace_id"))
        
        elif topic == "infercell.fork":
            self._realize_cell(data.get("child_trace_id"))
        
        elif topic == "infercell.merge":
            # For merge, we might want to archive sources or prep the merged workspace
            self._realize_cell(data.get("merged_trace_id"))


def main():
    parser = argparse.ArgumentParser(description="InferCell Activator Daemon")
    parser.add_argument("--root", default="/pluribus", help="Pluribus root")
    parser.add_argument("--bus-dir", default=None, help="Bus directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    bus_dir = Path(args.bus_dir).resolve() if args.bus_dir else (root / ".pluribus" / "bus")

    activator = InferCellActivator(root, bus_dir)
    activator.run()


if __name__ == "__main__":
    main()
