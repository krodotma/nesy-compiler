#!/usr/bin/env python3
"""
Dashboard Holistic Startup & Verification System
=================================================
Ensures all services are operational with recursive self-repair.

Services managed:
1. Bus Bridge (WS 9200, API 9201) - Event pub/sub
2. Git/FS Server (9300) - File browser + isomorphic-git
3. VPS Session Daemon - Provider monitoring
4. Omega Heartbeat - System liveness
5. Dashboard Dev Server (5173) - Qwik frontend

Usage:
    python3 dashboard_start.py [--verify-only] [--restart-all]
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
PLURIBUS_ROOT = Path("/pluribus")
PLURIBUS_BUS_DIR = PLURIBUS_ROOT / ".pluribus" / "bus"
DASHBOARD_DIR = PLURIBUS_ROOT / "nucleus" / "dashboard"
TOOLS_DIR = PLURIBUS_ROOT / "nucleus" / "tools"

SERVICES = {
    "bus_bridge": {
        "name": "Bus Bridge",
        "port": 9200,
        "api_port": 9201,
        "health_url": "http://localhost:9201/health",
        "start_cmd": ["npx", "tsx", "src/lib/bus/bus-bridge.ts"],
        "cwd": str(DASHBOARD_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR)},
        "critical": True,
        "startup_timeout": 10,
    },
    "git_server": {
        "name": "Git/FS Server",
        "port": 9300,
        "health_url": "http://localhost:9300/health",
        "health_timeout": 6.0,
        "start_cmd": ["python3", str(TOOLS_DIR / "git_server.py"), "--root", str(PLURIBUS_ROOT), "--port", "9300"],
        "cwd": str(TOOLS_DIR),
        "env": {},
        "critical": True,
        "startup_timeout": 10,
    },
    "dashboard_dev": {
        "name": "Dashboard Dev Server",
        "port": 5173,
        "health_url": "http://localhost:5173/",
        "health_timeout": 12.0,
        "start_cmd": ["npm", "run", "dev", "--", "--mode", "ssr", "--host", "0.0.0.0", "--port", "5173"],
        "cwd": str(DASHBOARD_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR)},
        "critical": True,
        "startup_timeout": 30,
        "post_start_sleep_s": 8,
    },
    "vps_session": {
        "name": "VPS Session Daemon",
        "port": None,
        "process_pattern": "vps_session.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "vps_session.py"), "daemon", "--interval-s", "300"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR)},
        "critical": True,
    },
    "omega_heartbeat": {
        "name": "Omega Heartbeat",
        "port": None,
        "process_pattern": "omega_heartbeat.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "omega_heartbeat.py")],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "OMEGA_INTERVAL_S": "10"},
        "critical": False,
        "startup_timeout": 10,
    },
    "dialogosd": {
        "name": "Dialogos Daemon",
        "port": None,
        "process_pattern": "dialogosd.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "dialogosd.py"), "--poll", "0.1"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "a2a_bridge": {
        "name": "A2A Bridge",
        "port": None,
        "process_pattern": "a2a_bridge.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "a2a_bridge.py"), "--poll", "0.2", "--providers", "auto"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "a2a_negotiate": {
        "name": "A2A Negotiate Daemon",
        "port": None,
        "process_pattern": "a2a/negotiate_daemon.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "a2a" / "negotiate_daemon.py"), "--poll", "0.25"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "pbflush_responder": {
        "name": "PBFLUSH Responder",
        "port": None,
        "process_pattern": "pbflush_responder.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "pbflush_responder.py"), "--poll", "0.25", "--since-ts", "0"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "rd_tasks_responder": {
        "name": "RD Tasks Responder",
        "port": None,
        "process_pattern": "rd_tasks_responder.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "rd_tasks_responder.py"), "--poll", "0.25", "--since-ts", "0"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "rd_tasks_worker": {
        "name": "RD Tasks Worker (optional exec)",
        "port": None,
        "process_pattern": "rd_tasks_worker.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "rd_tasks_worker.py"), "--poll", "0.25", "--since-ts", "0"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "mcp_host": {
        "name": "MCP Host (daemon)",
        "port": None,
        "process_pattern": "nucleus/mcp/host.py",
        "start_cmd": ["python3", str(PLURIBUS_ROOT / "nucleus" / "mcp" / "host.py"), "--root", str(PLURIBUS_ROOT), "daemon", "--bus-dir", str(PLURIBUS_BUS_DIR)],
        "cwd": str(PLURIBUS_ROOT / "nucleus" / "mcp"),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "mabswarm": {
        "name": "MABSWARM (daemon)",
        "port": None,
        "process_pattern": "mabswarm.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "mabswarm.py"), "--daemon", "--emit-bus"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "ui_backend": {
        "name": "UI Backend Daemon",
        "port": None,
        "process_pattern": "ui_backend_daemon.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "ui_backend_daemon.py"), "--interval", "10"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "infercell_activator": {
        "name": "InferCell Activator",
        "port": None,
        "process_pattern": "infercell_activator.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "infercell_activator.py")],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "catalog_daemon": {
        "name": "Catalog Daemon",
        "port": None,
        "process_pattern": "catalog_daemon.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "catalog_daemon.py"), "--interval", "15"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
    "pluribuscheck_responder": {
        "name": "PLURIBUSCHECK Responder",
        "port": None,
        "process_pattern": "pluribus_check_responder.py",
        "start_cmd": ["python3", str(TOOLS_DIR / "pluribus_check_responder.py"), "--poll", "0.25", "--since-ts", "0"],
        "cwd": str(TOOLS_DIR),
        "env": {"PLURIBUS_BUS_DIR": str(PLURIBUS_BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
        "critical": False,
        "startup_timeout": 10,
    },
}

MAX_REPAIR_ATTEMPTS = 3
STARTUP_TIMEOUT = 10
LOG_DIR = PLURIBUS_ROOT / ".pluribus" / "dashboard"


def log(msg: str, level: str = "INFO"):
    """Print timestamped log message."""
    ts = time.strftime("%H:%M:%S")
    symbol = {"INFO": "ℹ", "OK": "✓", "WARN": "⚠", "ERROR": "✗", "START": "▶"}.get(level, "•")
    print(f"[{ts}] {symbol} {msg}")


def check_port(port: int) -> bool:
    """Check if port is listening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("localhost", port)) == 0
    except:
        return False


def check_process(pattern: str) -> Optional[int]:
    """Check if process matching pattern is running, return PID."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split()[0])
    except:
        pass
    return None


def check_http_health(url: str, timeout: float = 3.0) -> Tuple[bool, str]:
    """Check HTTP health endpoint."""
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "*/*",
                "User-Agent": "pluribus-dashboard-start/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode('utf-8')
            return True, body
    except Exception as e:
        return False, str(e)


def kill_service(service_id: str, svc: dict) -> bool:
    """Kill a service by port or pattern."""
    killed = False

    # Kill by port
    if svc.get("port"):
        try:
            result = subprocess.run(
                ["fuser", "-k", f"{svc['port']}/tcp"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                killed = True
        except:
            pass
    if svc.get("api_port"):
        try:
            result = subprocess.run(
                ["fuser", "-k", f"{svc['api_port']}/tcp"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                killed = True
        except:
            pass

    # Kill by pattern
    if svc.get("process_pattern"):
        try:
            subprocess.run(
                ["pkill", "-f", svc["process_pattern"]],
                capture_output=True,
                timeout=5
            )
            killed = True
        except:
            pass

    time.sleep(1)
    return killed


def start_service(service_id: str, svc: dict) -> bool:
    """Start a service."""
    log(f"Starting {svc['name']}...", "START")

    env = {**os.environ, **svc.get("env", {})}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{service_id}.log"

    try:
        # Start in background
        with open(log_path, "ab", buffering=0) as log_fp:
            log_fp.write(f"\n\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} START {service_id} ===\n".encode())
            subprocess.Popen(
                svc["start_cmd"],
                cwd=svc.get("cwd"),
                env=env,
                stdout=log_fp,
                stderr=log_fp,
                start_new_session=True,
            )

        # Wait for startup
        timeout_s = int(svc.get("startup_timeout") or STARTUP_TIMEOUT)
        for i in range(timeout_s):
            time.sleep(1)

            # Check port
            if svc.get("port") and check_port(svc["port"]):
                log(f"{svc['name']} started (port {svc['port']})", "OK")
                return True

            # Check process
            if svc.get("process_pattern") and check_process(svc["process_pattern"]):
                log(f"{svc['name']} started (process running)", "OK")
                return True

        log(f"{svc['name']} failed to start within {timeout_s}s (see {log_path})", "ERROR")
        return False

    except Exception as e:
        log(f"{svc['name']} start failed: {e}", "ERROR")
        return False


def verify_service(service_id: str, svc: dict) -> Tuple[bool, str]:
    """Verify a service is healthy."""

    # Check port
    if svc.get("port"):
        if not check_port(svc["port"]):
            return False, f"Port {svc['port']} not listening"

    # Check process
    if svc.get("process_pattern"):
        if not check_process(svc["process_pattern"]):
            return False, f"Process not running"

    # Check HTTP health
    if svc.get("health_url"):
        ok, body = check_http_health(svc["health_url"], timeout=float(svc.get("health_timeout") or 3.0))
        if not ok:
            return False, f"Health check failed: {body}"

        # Parse JSON health response
        try:
            data = json.loads(body)
            if data.get("status") == "healthy":
                return True, f"Healthy (clients: {data.get('clients', '?')})"
        except:
            pass

        return True, "Responding"

    return True, "OK"


def repair_service(service_id: str, svc: dict, attempt: int) -> bool:
    """Attempt to repair a service."""
    log(f"Repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS} for {svc['name']}", "WARN")

    # Kill existing
    kill_service(service_id, svc)

    # Restart
    return start_service(service_id, svc)


def verify_all() -> Dict[str, Tuple[bool, str]]:
    """Verify all services."""
    results = {}
    for service_id, svc in SERVICES.items():
        ok, msg = verify_service(service_id, svc)
        results[service_id] = (ok, msg)
        status = "OK" if ok else "ERROR"
        log(f"{svc['name']}: {msg}", status)
    return results


def start_all(force_restart: bool = False) -> bool:
    """Start all services with recursive repair."""
    log("=" * 50)
    log("Dashboard Startup & Verification")
    log("=" * 50)

    all_ok = True

    for service_id, svc in SERVICES.items():
        log(f"\n--- {svc['name']} ---")

        # Check current status
        ok, msg = verify_service(service_id, svc)

        if ok and not force_restart:
            log(f"Already running: {msg}", "OK")
            continue

        if force_restart and ok:
            log("Force restart requested", "WARN")
            kill_service(service_id, svc)

        # Start with repair loop
        started = False
        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            if attempt > 1:
                started = repair_service(service_id, svc, attempt)
            else:
                started = start_service(service_id, svc)

            if started:
                # Verify after start
                time.sleep(float(svc.get("post_start_sleep_s", 2)))
                ok, msg = verify_service(service_id, svc)
                if ok:
                    log(f"Verified: {msg}", "OK")
                    break
                else:
                    log(f"Verification failed: {msg}", "WARN")
                    started = False

        if not started:
            log(f"Failed to start {svc['name']} after {MAX_REPAIR_ATTEMPTS} attempts", "ERROR")
            if svc.get("critical"):
                all_ok = False

    # Final verification
    log("\n" + "=" * 50)
    log("Final Verification")
    log("=" * 50)

    results = verify_all()

    failed = [k for k, (ok, _) in results.items() if not ok and SERVICES[k].get("critical")]

    if failed:
        log(f"\n✗ STARTUP FAILED - Critical services down: {failed}", "ERROR")
        return False
    else:
        log("\n✓ ALL SERVICES OPERATIONAL", "OK")
        return True


def emit_startup_event():
    """Emit startup event to bus."""
    event = {
        "id": f"startup-{int(time.time())}",
        "topic": "dashboard.startup",
        "kind": "log",
        "level": "info",
        "actor": "dashboard_start",
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": {
            "services": list(SERVICES.keys()),
            "status": "started"
        }
    }

    events_file = PLURIBUS_BUS_DIR / "events.ndjson"
    try:
        with open(events_file, "a") as f:
            f.write(json.dumps(event) + "\n")
        log("Emitted startup event to bus", "OK")
    except Exception as e:
        log(f"Failed to emit startup event: {e}", "WARN")


def main():
    parser = argparse.ArgumentParser(description="Dashboard Startup & Verification")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't start")
    parser.add_argument("--restart-all", action="store_true", help="Force restart all services")
    args = parser.parse_args()

    if args.verify_only:
        results = verify_all()
        failed = [k for k, (ok, _) in results.items() if not ok]
        sys.exit(1 if failed else 0)

    success = start_all(force_restart=args.restart_all)

    if success:
        emit_startup_event()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
