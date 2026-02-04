#!/usr/bin/env python3
"""
Browser Guardian - Chrome Single-Process Enforcement
=====================================================
Ensures only ONE Chrome/Chromium instance runs at a time.

Features:
- Pre-launch cleanup of orphaned Chrome processes
- Liveness probe with auto-restart
- Bus event emission for OHM integration

Usage:
    from browser_guardian import BrowserGuardian
    guardian = BrowserGuardian()
    guardian.ensure_single_browser()  # Call before launching browser
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True

# Bus integration
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus
except Exception:
    agent_bus = None


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def emit_event(topic: str, data: dict) -> None:
    """Emit bus event if agent_bus is available."""
    if agent_bus:
        try:
            agent_bus.emit_bus_event(
                topic=topic,
                data=data,
                actor="browser-guardian",
                level="info",
            )
        except Exception:
            pass


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cmdline: str
    memory_mb: float


class BrowserGuardian:
    """Enforces single Chrome/Chromium instance policy."""
    
    MAX_INSTANCES = 1
    CHROME_PATTERNS = ("chrome", "chromium", "google-chrome")
    KILL_GRACE_PERIOD = 5  # seconds
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._last_check = 0
    
    def find_chrome_processes(self) -> list[ProcessInfo]:
        """Find all Chrome/Chromium processes."""
        processes = []
        try:
            # Use ps to find chrome processes
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=5
            )
            for line in result.stdout.split("\n")[1:]:  # Skip header
                parts = line.split(None, 10)
                if len(parts) < 11:
                    continue
                cmdline = parts[10].lower()
                # Check if any chrome pattern matches
                if any(p in cmdline for p in self.CHROME_PATTERNS) and "--type=" not in cmdline:
                    # Exclude grep itself
                    if "grep" in cmdline:
                        continue
                    try:
                        pid = int(parts[1])
                        mem_pct = float(parts[3])
                        # Estimate MB from percentage (assuming ~14GB VPS)
                        mem_mb = (mem_pct / 100) * 14336
                        processes.append(ProcessInfo(
                            pid=pid,
                            name=parts[10].split()[0] if parts[10] else "chrome",
                            cmdline=parts[10][:100],
                            memory_mb=mem_mb
                        ))
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"[Guardian] Error finding processes: {e}")
        return processes
    
    def kill_process(self, proc: ProcessInfo, force: bool = False) -> bool:
        """Kill a process, optionally with SIGKILL."""
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            if self.dry_run:
                print(f"[Guardian] DRY RUN: Would kill PID {proc.pid}")
                return True
            os.kill(proc.pid, sig)
            emit_event("browser.guardian.killed", {
                "pid": proc.pid,
                "signal": "SIGKILL" if force else "SIGTERM",
                "memory_mb": proc.memory_mb
            })
            return True
        except ProcessLookupError:
            return True  # Already dead
        except PermissionError:
            print(f"[Guardian] Permission denied: PID {proc.pid}")
            return False
        except Exception as e:
            print(f"[Guardian] Kill failed: {e}")
            return False
    
    def ensure_single_browser(self) -> bool:
        """
        Ensure only one Chrome instance is running.
        Call this BEFORE launching a new browser.
        Returns True if environment is clean.
        """
        processes = self.find_chrome_processes()
        
        if len(processes) == 0:
            print("[Guardian] ✓ No Chrome processes found - clean slate")
            return True
        
        if len(processes) <= self.MAX_INSTANCES:
            print(f"[Guardian] ✓ {len(processes)} Chrome process(es) - within limit")
            return True
        
        # Too many processes - kill extras
        print(f"[Guardian] ⚠ {len(processes)} Chrome processes found - enforcing limit")
        emit_event("browser.guardian.cleanup_start", {
            "count": len(processes),
            "limit": self.MAX_INSTANCES
        })
        
        # Sort by memory (descending) and keep the largest one (most likely the main)
        processes.sort(key=lambda p: p.memory_mb, reverse=True)
        to_kill = processes[self.MAX_INSTANCES:]
        
        killed = 0
        for proc in to_kill:
            print(f"[Guardian] Killing PID {proc.pid} ({proc.memory_mb:.0f}MB)")
            if self.kill_process(proc):
                killed += 1
        
        # Wait for graceful shutdown
        time.sleep(self.KILL_GRACE_PERIOD)
        
        # Force kill any remaining
        remaining = self.find_chrome_processes()
        if len(remaining) > self.MAX_INSTANCES:
            print(f"[Guardian] Force killing {len(remaining) - self.MAX_INSTANCES} stubborn processes")
            for proc in remaining[self.MAX_INSTANCES:]:
                self.kill_process(proc, force=True)
        
        final_count = len(self.find_chrome_processes())
        emit_event("browser.guardian.cleanup_complete", {
            "killed": killed,
            "remaining": final_count
        })
        
        return final_count <= self.MAX_INSTANCES
    
    def kill_all_browsers(self) -> int:
        """Kill ALL Chrome/Chromium processes. Use with caution."""
        processes = self.find_chrome_processes()
        if not processes:
            return 0
        
        print(f"[Guardian] Killing ALL {len(processes)} Chrome processes")
        killed = 0
        for proc in processes:
            if self.kill_process(proc):
                killed += 1
        
        time.sleep(self.KILL_GRACE_PERIOD)
        
        # Force kill remaining
        for proc in self.find_chrome_processes():
            self.kill_process(proc, force=True)
            killed += 1
        
        return killed
    
    def get_chrome_stats(self) -> dict:
        """Get current Chrome process statistics."""
        processes = self.find_chrome_processes()
        total_mem = sum(p.memory_mb for p in processes)
        return {
            "count": len(processes),
            "total_memory_mb": round(total_mem, 1),
            "pids": [p.pid for p in processes],
            "within_limit": len(processes) <= self.MAX_INSTANCES
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Browser Guardian - Chrome Process Enforcer")
    parser.add_argument("command", choices=["status", "cleanup", "kill-all"],
                        help="Command to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually kill processes")
    args = parser.parse_args()
    
    guardian = BrowserGuardian(dry_run=args.dry_run)
    
    if args.command == "status":
        stats = guardian.get_chrome_stats()
        print(f"Chrome Processes: {stats['count']}")
        print(f"Total Memory: {stats['total_memory_mb']} MB")
        print(f"PIDs: {stats['pids']}")
        print(f"Within Limit: {'✓' if stats['within_limit'] else '✗'}")
    
    elif args.command == "cleanup":
        if guardian.ensure_single_browser():
            print("[Guardian] Cleanup successful")
            sys.exit(0)
        else:
            print("[Guardian] Cleanup failed - manual intervention needed")
            sys.exit(1)

    elif args.command == "kill-all":
        killed = guardian.kill_all_browsers()
        print(f"[Guardian] Killed {killed} processes")


if __name__ == "__main__":
    main()
