#!/usr/bin/env python3
"""
Bus Consumer Core Library
=========================

Unified utility for efficiently consuming events from the Pluribus bus.
Supports:
1. Grep-based pre-filtering for cold logs.
2. Seek-based tailing for hot logs.
3. Batching and offset management.
4. Topic filtering.

Compliance: Sextet 'L' (Local file access only)
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional, Iterator, Any

class BusConsumer:
    """Unified consumer for the Pluribus event bus."""

    def __init__(self, bus_path: Path, state_path: Optional[Path] = None):
        self.bus_path = Path(bus_path)
        self.state_path = Path(state_path) if state_path else None
        self._last_id: Optional[str] = None
        self._last_offset: int = 0
        self.load_state()

    def load_state(self):
        """Load last processed event ID and offset from state file."""
        if self.state_path and self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text())
                self._last_id = state.get("last_id")
                self._last_offset = state.get("last_offset", 0)
            except Exception:
                pass

    def save_state(self):
        """Save last processed event ID and offset."""
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "last_id": self._last_id,
                "last_offset": self._last_offset,
                "ts": time.time()
            }
            self.state_path.write_text(json.dumps(state))

    def _grep_stream(self, pattern: str) -> Iterator[dict]:
        """Stream events matching a regex pattern using grep."""
        grep_cmd = ["grep", "-E", pattern, str(self.bus_path)]
        try:
            with subprocess.Popen(grep_cmd, stdout=subprocess.PIPE, text=True, bufsize=1) as proc:
                if proc.stdout:
                    for line in proc.stdout:
                        if not line.strip():
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            # Fallback to standard read if grep fails
            with self.bus_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if pattern in line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

    def scan(self, topic_filter: Optional[str] = None, callback: Optional[Callable[[dict], Any]] = None) -> int:
        """Scan the entire bus (or from last state) and process events."""
        count = 0
        pattern = topic_filter if topic_filter else "."
        
        # If we have a last_id, we should ideally skip until we find it.
        # However, for massive files, seek-to-offset is better if we have it.
        found_last = (self._last_id is None)
        
        for event in self._grep_stream(pattern):
            eid = event.get("id")
            if not found_last:
                if eid == self._last_id:
                    found_last = True
                continue
            
            if callback:
                callback(event)
            
            self._last_id = eid
            count += 1
            if count % 1000 == 0:
                self.save_state()
        
        self.save_state()
        return count

    def tail(self, topic_filter: Optional[str] = None, callback: Optional[Callable[[dict], Any]] = None, idle_callback: Optional[Callable[[], Any]] = None, poll_s: float = 0.5):
        """Tail the bus for new events, handling log rotation."""
        print(f"[BusConsumer] tailing {self.bus_path} (filter: {topic_filter or 'none'})")
        
        def open_file():
            h = self.bus_path.open("r", encoding="utf-8", errors="replace")
            h.seek(0, os.SEEK_END)
            ino = os.fstat(h.fileno()).st_ino
            return h, ino

        handle, inode = open_file()
        self._last_offset = handle.tell()
        
        while True:
            line = handle.readline()
            if not line:
                if idle_callback:
                    idle_callback()

                # Check for rotation
                try:
                    stats = self.bus_path.stat()
                    if stats.st_ino != inode:
                        print(f"[BusConsumer] rotation detected for {self.bus_path}")
                        handle.close()
                        handle, inode = open_file()
                        continue
                except FileNotFoundError:
                    pass
                
                time.sleep(poll_s)
                continue
            
            self._last_offset = handle.tell()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if not isinstance(event, dict):
                continue
            
            # Simple topic filter check
            if topic_filter:
                topic = event.get("topic", "")
                if not any(t.strip() in topic for t in topic_filter.split("|")):
                    continue
            
            if callback:
                callback(event)
            
            self._last_id = event.get("id")

if __name__ == "__main__":
    # Quick test
    bus = Path("/pluribus/.pluribus/bus/events.ndjson")
    if bus.exists():
        consumer = BusConsumer(bus)
        print("Scanning last 5 events...")
        # (This is just a demo, real usage would be via library)
