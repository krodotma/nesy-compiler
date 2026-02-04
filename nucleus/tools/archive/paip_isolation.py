#!/usr/bin/env python3
"""
PAIP v13.1 Isolation Manager
==========================
Calculates and enforces Phenomenological Isolation (Ports, Displays, Cache Dirs)
for parallel agents and environment tiers (Prod, Staging, Dev).

Usage:
    # Get environment exports for current agent/clone
    python3 nucleus/tools/paip_isolation.py env

    # Reap processes for a specific slot
    python3 nucleus/tools/paip_isolation.py reap --id dev
"""

import argparse
import hashlib
import os
import socket
import sys
import subprocess
import signal
from pathlib import Path

# Base configuration
BASE_PORT_DASHBOARD = 5173
BASE_PORT_API = 8080
BASE_DISPLAY = 99  # Start Xvfb displays at :99 to avoid :0 collision

# Tier Configuration (Fixed Slots)
TIER_CONFIG = {
    "prod": {"port": 5173, "api": 8080, "display": ":99", "slot": 0, "host": "kroma.live"},
    "staging": {"port": 5180, "api": 8081, "display": ":100", "slot": 1, "host": "staging.kroma.live"},
    "dev": {"port": 5190, "api": 8082, "display": ":101", "slot": 2, "host": "dev.kroma.live"},
}

def get_clone_id(explicit_id=None) -> str:
    """Derive unique ID from args, env, CWD, or fallback."""
    if explicit_id:
        return explicit_id
    
    # Priority 1: Explicit PAIP ID env var
    if os.environ.get("PLURIBUS_PAIP_ID"):
        return os.environ["PLURIBUS_PAIP_ID"]
    
    # Priority 2: Infer from CWD if it looks like a clone
    cwd = os.getcwd()
    if "/tmp/pluribus_" in cwd:
        return Path(cwd).name
    
    # Priority 3: Fallback to username/actor
    return os.environ.get("USER", "unknown")

def get_isolation_slot(clone_id: str) -> int:
    """Deterministic slot (0-50) based on clone ID hash."""
    if clone_id in TIER_CONFIG:
        return TIER_CONFIG[clone_id]["slot"]
    if clone_id == "unknown":
        return 0
    # Dynamic slots start at 10 to avoid TIER collision (0, 1, 2)
    return 10 + (int(hashlib.md5(clone_id.encode()).hexdigest(), 16) % 40)

def get_isolated_config(clone_id_arg=None) -> dict:
    clone_id = get_clone_id(clone_id_arg)
    
    # Check for TIER override
    if clone_id in TIER_CONFIG:
        t = TIER_CONFIG[clone_id]
        return {
            "PAIP_ID": clone_id,
            "PAIP_SLOT": str(t["slot"]),
            "PORT": str(t["port"]),
            "API_PORT": str(t["api"]),
            "DISPLAY": t["display"],
            "VITE_PORT": str(t["port"]),
            "HOST": t["host"],
        }

    slot = get_isolation_slot(clone_id)
    
    # Shift ports for dynamic slots
    port_dashboard = BASE_PORT_DASHBOARD + slot
    port_api = BASE_PORT_API + slot
    display = f":{BASE_DISPLAY + slot}"
    host = f"slot-{slot}.kroma.live"
    
    return {
        "PAIP_ID": clone_id,
        "PAIP_SLOT": str(slot),
        "PORT": str(port_dashboard),
        "API_PORT": str(port_api),
        "DISPLAY": display,
        "VITE_PORT": str(port_dashboard),
        "HOST": host,
    }

def get_pids_on_port(port: int) -> list[int]:
    """Find PIDs listening on a port."""
    try:
        # Use lsof to find listening PIDs
        out = subprocess.check_output(["lsof", "-t", f"-i:{port}"], stderr=subprocess.DEVNULL)
        return [int(p) for p in out.decode().strip().split()]
    except Exception:
        return []

def get_pids_on_display(display: str) -> list[int]:
    """Find Xvfb or X servers on a display."""
    try:
        out = subprocess.check_output(["pgrep", "-f", f"X.*{display}"], stderr=subprocess.DEVNULL)
        return [int(p) for p in out.decode().strip().split()]
    except Exception:
        return []

def kill_pid(pid: int):
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"  -> Killed PID {pid}")
    except ProcessLookupError:
        pass
    except Exception as e:
        print(f"  -> Failed to kill {pid}: {e}")

def reap_slot(clone_id: str):
    print(f"Reaping PAIP resources for ID: {clone_id} ...")
    config = get_isolated_config(clone_id)
    
    # 1. Dashboard Port
    dash_port = int(config["PORT"])
    pids = get_pids_on_port(dash_port)
    if pids:
        print(f"  Found dashboard processes on port {dash_port}: {pids}")
        for p in pids: kill_pid(p)
        
    # 2. API Port
    api_port = int(config["API_PORT"])
    pids = get_pids_on_port(api_port)
    if pids:
        print(f"  Found API processes on port {api_port}: {pids}")
        for p in pids: kill_pid(p)

    # 3. Display
    display = config["DISPLAY"]
    pids = get_pids_on_display(display)
    if pids:
        print(f"  Found X/Display processes on {display}: {pids}")
        for p in pids: kill_pid(p)
        
    print(f"Reap complete for {clone_id} (Slot {config['PAIP_SLOT']})")

def print_env(clone_id=None):
    config = get_isolated_config(clone_id)
    print(f"# PAIP v13.1 Isolation for {config['PAIP_ID']} (Slot {config['PAIP_SLOT']})")
    for k, v in config.items():
        print(f"export {k}={v}")

def main():
    parser = argparse.ArgumentParser(description="PAIP v13.1 Isolation Manager")
    parser.add_argument("command", choices=["env", "check", "reap"], default="env")
    parser.add_argument("--id", help="Explicit Clone/PAIP ID or TIER (prod|staging|dev)")
    args = parser.parse_args()
    
    if args.command == "env":
        print_env(args.id)
    elif args.command == "check":
        cfg = get_isolated_config(args.id)
        print(f"Clone: {cfg['PAIP_ID']}")
        print(f"Slot: {cfg['PAIP_SLOT']}")
        print(f"Dashboard Port: {cfg['PORT']}")
        print(f"Display: {cfg['DISPLAY']}")
    elif args.command == "reap":
        target_id = get_clone_id(args.id)
        reap_slot(target_id)

if __name__ == "__main__":
    main()