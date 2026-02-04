#!/usr/bin/env python3
"""Dashboard Service Supervisor.

Manages always-on services for the Pluribus dashboard:
- VPS Session Daemon (provider monitoring, flow mode)
- Dashboard Bridge (WebSocket for web/native clients)
- Native Dashboard (optional, for terminal display)

Features:
- Auto-restart on failure
- Live reload on config changes
- Bus-controlled start/stop/restart
- TUI/Web configurable

Usage:
    python3 dashboard_supervisor.py start           # Start all services
    python3 dashboard_supervisor.py stop            # Stop all services
    python3 dashboard_supervisor.py status          # Show service status
    python3 dashboard_supervisor.py restart vps     # Restart specific service
    python3 dashboard_supervisor.py daemon          # Run as supervisor daemon

Bus Commands (emit to control):
    dashboard.supervisor.start   {"service": "vps"|"bridge"|"all"}
    dashboard.supervisor.stop    {"service": "vps"|"bridge"|"all"}
    dashboard.supervisor.restart {"service": "vps"|"bridge"|"all"}
    dashboard.supervisor.reload  {}  # Reload config
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


@dataclass
class ServiceConfig:
    """Configuration for a managed service."""
    id: str
    name: str
    enabled: bool = True
    auto_restart: bool = True
    restart_delay: float = 2.0
    max_restarts: int = 10
    restart_window: float = 60.0  # Reset restart count after this many seconds


@dataclass
class ServiceState:
    """Runtime state of a managed service."""
    id: str
    pid: int | None = None
    status: str = "stopped"  # stopped, starting, running, failed
    restart_count: int = 0
    last_restart: float = 0.0
    last_error: str | None = None
    started_iso: str = ""


@dataclass
class SupervisorConfig:
    """Supervisor configuration."""
    vps_enabled: bool = True
    vps_interval_s: int = 30
    bridge_enabled: bool = True
    bridge_port: int = 9200
    auto_start: bool = True
    live_reload: bool = True


class DashboardSupervisor:
    """Manages dashboard services with TIER awareness."""

    CONFIG_FILE = "dashboard_supervisor.json"

    def __init__(self, root: Path, tier: str = "prod"):
        self.root = root
        self.tier = tier
        self.pluribus_dir = root / ".pluribus"
        self.config_path = self.pluribus_dir / f"{tier}_{self.CONFIG_FILE}"
        self.tools_dir = root / "nucleus" / "tools"
        self.bus_dir = self.pluribus_dir / "bus"
        self.events_path = self.bus_dir / "events.ndjson"

        # Load isolation config for this tier
        if str(self.tools_dir) not in sys.path:
            sys.path.append(str(self.tools_dir))
        try:
            import paip_isolation
            self.iso_config = paip_isolation.get_isolated_config(tier)
        except ImportError:
            self.iso_config = {
                "PORT": "5173", 
                "API_PORT": "8080", 
                "DISPLAY": ":99", 
                "PAIP_SLOT": "0"
            }

        self.config = self._load_config()
        # Use isolated bridge port based on slot: 9200 + slot
        bridge_port = 9200 + int(self.iso_config.get("PAIP_SLOT", "0"))
        
        self.services: dict[str, ServiceConfig] = {
            "vps": ServiceConfig(
                id="vps",
                name="VPS Session Daemon",
                enabled=self.config.vps_enabled,
            ),
            "bridge": ServiceConfig(
                id="bridge",
                name="Dashboard Bridge",
                enabled=self.config.bridge_enabled,
            ),
            "vite": ServiceConfig(
                id="vite",
                name="Vite Dev Server",
                enabled=True,
            ),
            "dialogos": ServiceConfig(
                id="dialogos",
                name="Dialogos Daemon",
                enabled=True,
                restart_delay=2.0,
            ),
            "a2a-bridge": ServiceConfig(
                id="a2a-bridge",
                name="A2A Bridge",
                enabled=True,
                restart_delay=2.0,
            ),
            "catalog": ServiceConfig(
                id="catalog",
                name="Catalog Daemon",
                enabled=True,
                restart_delay=5.0,
            ),
            "git": ServiceConfig(
                id="git",
                name="Git/FS Server",
                enabled=True,
            ),
        }
        self.states: dict[str, ServiceState] = {
            "vps": ServiceState(id="vps"),
            "bridge": ServiceState(id="bridge"),
            "vite": ServiceState(id="vite"),
            "dialogos": ServiceState(id="dialogos"),
            "a2a-bridge": ServiceState(id="a2a-bridge"),
            "catalog": ServiceState(id="catalog"),
            "git": ServiceState(id="git"),
        }
        self.processes: dict[str, subprocess.Popen] = {}
        self.running = False
        self.config_mtime = 0.0

    def _load_config(self) -> SupervisorConfig:
        """Load config from disk."""
        if self.config_path.exists():
            try:
                with self.config_path.open() as f:
                    data = json.load(f)
                return SupervisorConfig(
                    vps_enabled=data.get("vps_enabled", True),
                    vps_interval_s=data.get("vps_interval_s", 30),
                    bridge_enabled=data.get("bridge_enabled", True),
                    bridge_port=data.get("bridge_port", 9200 + int(self.iso_config.get("PAIP_SLOT", "0"))),
                    auto_start=data.get("auto_start", True),
                    live_reload=data.get("live_reload", True),
                )
            except Exception:
                pass
        
        # Default with tier offset
        default_bridge = 9200 + int(self.iso_config.get("PAIP_SLOT", "0"))
        return SupervisorConfig(bridge_port=default_bridge)

    def save_config(self) -> None:
        """Save config to disk."""
        ensure_dir(self.config_path.parent)
        with self.config_path.open("w") as f:
            json.dump(asdict(self.config), f, indent=2)
        self._emit_bus(f"dashboard.supervisor.{self.tier}.config_saved", asdict(self.config))

    def _emit_bus(self, topic: str, data: dict) -> None:
        """Emit event to bus."""
        bus_tool = self.tools_dir / "agent_bus.py"
        if not bus_tool.exists():
            return
        try:
            env = os.environ.copy()
            env["PLURIBUS_BUS_DIR"] = str(self.bus_dir)
            env["PLURIBUS_TIER"] = self.tier
            subprocess.run(
                [
                    sys.executable, str(bus_tool),
                    "pub",
                    "--topic", topic,
                    "--kind", "event",
                    "--level", "info",
                    "--actor", f"supervisor-{self.tier}",
                    "--data", json.dumps(data),
                ],
                env=env,
                check=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    def start_service(self, service_id: str) -> bool:
        """Start a service."""
        if service_id not in self.services:
            return False

        svc = self.services[service_id]
        state = self.states[service_id]

        if state.status == "running" and state.pid:
            # Already running
            return True

        state.status = "starting"
        state.started_iso = now_iso()

        # Shared environment for this tier
        env = os.environ.copy()
        env["PLURIBUS_BUS_DIR"] = str(self.bus_dir)
        env["PLURIBUS_TIER"] = self.tier
        env["PORT"] = self.iso_config["PORT"]
        env["VITE_PORT"] = self.iso_config["PORT"]
        env["DISPLAY"] = self.iso_config["DISPLAY"]

        try:
            if service_id == "vps":
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(self.tools_dir / "vps_session.py"),
                        "daemon",
                        "--interval-s", str(self.config.vps_interval_s),
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif service_id == "vite":
                dashboard_dir = self.root / "nucleus" / "dashboard"
                proc = subprocess.Popen(
                    ["npm", "run", "dev", "--", "--port", self.iso_config["PORT"], "--host", "0.0.0.0", "--strictPort"],
                    cwd=str(dashboard_dir),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif service_id == "bridge":
                dashboard_dir = self.root / "nucleus" / "dashboard"
                bridge_script = dashboard_dir / "src" / "lib" / "bus" / "bus-bridge.ts"
                if bridge_script.exists():
                    proc = subprocess.Popen(
                        ["npx", "tsx", str(bridge_script)],
                        cwd=str(dashboard_dir),
                        env={
                            **env,
                            "BUS_BRIDGE_WS_PORT": str(self.config.bridge_port),
                        },
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    state.status = "failed"
                    state.last_error = "bridge script not found"
                    return False
            elif service_id == "catalog":
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(self.tools_dir / "catalog_daemon.py"),
                        "--interval-s", "5",
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif service_id == "dialogos":
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(self.tools_dir / "dialogosd.py"),
                        "--bus-dir", str(self.bus_dir),
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif service_id == "a2a-bridge":
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(self.tools_dir / "a2a_bridge.py"),
                        "--bus-dir", str(self.bus_dir),
                    ],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif service_id == "git":
                # Offset git port by slot: 9300 + slot
                git_port = 9300 + int(self.iso_config.get("PAIP_SLOT", "0"))
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        str(self.tools_dir / "git_server.py"),
                        "--root", str(self.root),
                        "--port", str(git_port),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                return False

            self.processes[service_id] = proc
            state.pid = proc.pid
            state.status = "running"
            state.last_error = None

            self._emit_bus(f"dashboard.supervisor.{self.tier}.started", {
                "service": service_id,
                "pid": proc.pid,
                "iso": now_iso(),
            })
            return True

        except Exception as e:
            state.status = "failed"
            state.last_error = str(e)
            self._emit_bus(f"dashboard.supervisor.{self.tier}.failed", {
                "service": service_id,
                "error": str(e),
                "iso": now_iso(),
            })
            return False

    def stop_service(self, service_id: str) -> bool:
        """Stop a service."""
        if service_id not in self.services:
            return False

        state = self.states[service_id]
        proc = self.processes.get(service_id)

        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except Exception:
                pass
            del self.processes[service_id]

        state.status = "stopped"
        state.pid = None

        self._emit_bus("dashboard.supervisor.stopped", {
            "service": service_id,
            "iso": now_iso(),
        })
        return True

    def restart_service(self, service_id: str) -> bool:
        """Restart a service."""
        self.stop_service(service_id)
        time.sleep(0.5)
        return self.start_service(service_id)

    def check_services(self) -> None:
        """Check service health and restart if needed."""
        for svc_id, svc in self.services.items():
            if not svc.enabled:
                continue

            state = self.states[svc_id]
            proc = self.processes.get(svc_id)

            # Check if process died
            if proc and proc.poll() is not None:
                exit_code = proc.returncode
                state.status = "failed"
                state.last_error = f"exited with code {exit_code}"
                del self.processes[svc_id]

                # Auto-restart logic
                if svc.auto_restart:
                    now = time.time()
                    if now - state.last_restart > svc.restart_window:
                        state.restart_count = 0

                    if state.restart_count < svc.max_restarts:
                        state.restart_count += 1
                        state.last_restart = now
                        time.sleep(svc.restart_delay)
                        self.start_service(svc_id)
                        self._emit_bus("dashboard.supervisor.auto_restarted", {
                            "service": svc_id,
                            "restart_count": state.restart_count,
                            "iso": now_iso(),
                        })

            # Start if enabled but not running
            elif state.status == "stopped" and svc.enabled:
                self.start_service(svc_id)

    def reload_config(self) -> None:
        """Reload config from disk."""
        old_config = self.config
        self.config = self._load_config()

        # Apply changes
        self.services["vps"].enabled = self.config.vps_enabled
        self.services["bridge"].enabled = self.config.bridge_enabled

        # Restart services if config changed
        if self.config.vps_interval_s != old_config.vps_interval_s:
            self.restart_service("vps")
        if self.config.bridge_port != old_config.bridge_port:
            self.restart_service("bridge")

        self._emit_bus("dashboard.supervisor.config_reloaded", asdict(self.config))

    def check_config_changed(self) -> bool:
        """Check if config file changed (for live reload)."""
        if not self.config_path.exists():
            return False
        try:
            mtime = self.config_path.stat().st_mtime
            if mtime > self.config_mtime:
                self.config_mtime = mtime
                return True
        except Exception:
            pass
        return False

    def handle_bus_command(self, topic: str, data: dict) -> None:
        """Handle bus control commands."""
        service = data.get("service", "all")

        if topic == "dashboard.supervisor.start":
            if service == "all":
                for svc_id in self.services:
                    self.start_service(svc_id)
            else:
                self.start_service(service)

        elif topic == "dashboard.supervisor.stop":
            if service == "all":
                for svc_id in self.services:
                    self.stop_service(svc_id)
            else:
                self.stop_service(service)

        elif topic == "dashboard.supervisor.restart":
            if service == "all":
                for svc_id in self.services:
                    self.restart_service(svc_id)
            else:
                self.restart_service(service)

        elif topic == "dashboard.supervisor.reload":
            self.reload_config()

        elif topic == "dashboard.supervisor.set_config":
            # Update config from bus
            if "vps_enabled" in data:
                self.config.vps_enabled = bool(data["vps_enabled"])
            if "bridge_enabled" in data:
                self.config.bridge_enabled = bool(data["bridge_enabled"])
            if "vps_interval_s" in data:
                self.config.vps_interval_s = int(data["vps_interval_s"])
            if "bridge_port" in data:
                self.config.bridge_port = int(data["bridge_port"])
            self.save_config()
            self.reload_config()

    def run_daemon(self) -> None:
        """Run as supervisor daemon."""
        self.running = True
        self._emit_bus("dashboard.supervisor.daemon_started", {"iso": now_iso()})

        # Start enabled services
        if self.config.auto_start:
            for svc_id, svc in self.services.items():
                if svc.enabled:
                    self.start_service(svc_id)

        # Initialize config mtime for live reload
        if self.config_path.exists():
            self.config_mtime = self.config_path.stat().st_mtime

        # Watch bus for commands
        ensure_dir(self.events_path.parent)
        if not self.events_path.exists():
            self.events_path.write_text("", encoding="utf-8")

        def signal_handler(signum, frame):
            self.running = False

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        with self.events_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)

            while self.running:
                # Check services health
                self.check_services()

                # Live reload config
                if self.config.live_reload and self.check_config_changed():
                    self.reload_config()

                # Read bus commands
                line = f.readline()
                if line:
                    try:
                        obj = json.loads(line)
                        topic = obj.get("topic", "")
                        if topic.startswith("dashboard.supervisor."):
                            data = obj.get("data", {})
                            self.handle_bus_command(topic, data)
                    except Exception:
                        pass
                else:
                    time.sleep(0.5)

        # Cleanup
        for svc_id in list(self.processes.keys()):
            self.stop_service(svc_id)

        self._emit_bus("dashboard.supervisor.daemon_stopped", {"iso": now_iso()})

    def get_status(self) -> dict:
        """Get status of all services."""
        return {
            "config": asdict(self.config),
            "services": {
                svc_id: {
                    "config": asdict(self.services[svc_id]),
                    "state": asdict(self.states[svc_id]),
                }
                for svc_id in self.services
            },
        }


def cmd_start(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    sup = DashboardSupervisor(root, tier=args.tier)

    service = args.service if hasattr(args, "service") else "all"
    if service == "all":
        for svc_id in sup.services:
            if sup.start_service(svc_id):
                print(f"Started {svc_id} ({args.tier})")
            else:
                print(f"Failed to start {svc_id} ({args.tier})")
    else:
        if sup.start_service(service):
            print(f"Started {service} ({args.tier})")
        else:
            print(f"Failed to start {service} ({args.tier})")
            return 1
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    sup = DashboardSupervisor(root, tier=args.tier)

    service = args.service if hasattr(args, "service") else "all"
    if service == "all":
        for svc_id in sup.services:
            sup.stop_service(svc_id)
            print(f"Stopped {svc_id} ({args.tier})")
    else:
        sup.stop_service(service)
        print(f"Stopped {service} ({args.tier})")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    sup = DashboardSupervisor(root, tier=args.tier)

    service = args.service if hasattr(args, "service") else "all"
    if service == "all":
        for svc_id in sup.services:
            if sup.restart_service(svc_id):
                print(f"Restarted {svc_id} ({args.tier})")
            else:
                print(f"Failed to restart {svc_id} ({args.tier})")
    else:
        if sup.restart_service(service):
            print(f"Restarted {service} ({args.tier})")
        else:
            print(f"Failed to restart {service} ({args.tier})")
            return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    sup = DashboardSupervisor(root, tier=args.tier)
    status = sup.get_status()

    print(f"Dashboard Supervisor Status - TIER: {args.tier.upper()}")
    print("=" * 50)
    print(f"Live Reload: {'enabled' if status['config']['live_reload'] else 'disabled'}")
    print(f"Auto Start:  {'enabled' if status['config']['auto_start'] else 'disabled'}")
    print()

    for svc_id, svc_status in status["services"].items():
        cfg = svc_status["config"]
        state = svc_status["state"]
        enabled = "[+]" if cfg["enabled"] else "[-]"
        running = state["status"]
        pid = f"(PID: {state['pid']})" if state["pid"] else ""
        print(f"  {enabled} {cfg['name']}: {running} {pid}")
        if state["last_error"]:
            print(f"      Error: {state['last_error']}")

    if args.json:
        print()
        print(json.dumps(status, indent=2))

    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    sup = DashboardSupervisor(root, tier=args.tier)
    print(f"Starting dashboard supervisor daemon - TIER: {args.tier.upper()}")
    print(f"  VPS Daemon: {'enabled' if sup.config.vps_enabled else 'disabled'}")
    print(f"  Bridge:     {'enabled' if sup.config.bridge_enabled else 'disabled'}")
    print(f"  Live Reload: {'enabled' if sup.config.live_reload else 'disabled'}")
    print()
    print("Press Ctrl+C to stop")
    sup.run_daemon()
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    sup = DashboardSupervisor(root, tier=args.tier)

    if args.set:
        key, value = args.set.split("=", 1)
        if key == "vps_enabled":
            sup.config.vps_enabled = value.lower() in ("true", "1", "yes")
        elif key == "bridge_enabled":
            sup.config.bridge_enabled = value.lower() in ("true", "1", "yes")
        elif key == "vps_interval_s":
            sup.config.vps_interval_s = int(value)
        elif key == "bridge_port":
            sup.config.bridge_port = int(value)
        elif key == "auto_start":
            sup.config.auto_start = value.lower() in ("true", "1", "yes")
        elif key == "live_reload":
            sup.config.live_reload = value.lower() in ("true", "1", "yes")
        else:
            print(f"Unknown config key: {key}")
            return 1
        sup.save_config()
        print(f"Set {key}={value} for {args.tier}")
    else:
        print(json.dumps(asdict(sup.config), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dashboard Service Supervisor")
    parser.add_argument("--root", help="Pluribus root directory")
    parser.add_argument("--tier", choices=["prod", "staging", "dev"], default="prod", help="Target tier environment")

    subparsers = parser.add_subparsers(dest="command")

    # start
    start_p = subparsers.add_parser("start", help="Start services")
    start_p.add_argument("service", nargs="?", default="all", help="Service to start (vps, bridge, vite, all)")

    # stop
    stop_p = subparsers.add_parser("stop", help="Stop services")
    stop_p.add_argument("service", nargs="?", default="all", help="Service to stop")

    # restart
    restart_p = subparsers.add_parser("restart", help="Restart services")
    restart_p.add_argument("service", nargs="?", default="all", help="Service to restart")

    # status
    status_p = subparsers.add_parser("status", help="Show service status")
    status_p.add_argument("--json", action="store_true", help="Output as JSON")

    # daemon
    subparsers.add_parser("daemon", help="Run as supervisor daemon")

    # config
    config_p = subparsers.add_parser("config", help="View/set configuration")
    config_p.add_argument("--set", help="Set config value (key=value)")

    args = parser.parse_args()

    if not args.command:
        args = parser.parse_args(["status", "--tier", args.tier])

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "restart": cmd_restart,
        "status": cmd_status,
        "daemon": cmd_daemon,
        "config": cmd_config,
    }

    return commands.get(args.command, cmd_status)(args)


if __name__ == "__main__":
    sys.exit(main())
