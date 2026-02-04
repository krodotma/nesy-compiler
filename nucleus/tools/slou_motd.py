#!/usr/bin/env python3
"""
SLOU: System Load Ontological Unit (The "SuperMOTD" Generator)
==============================================================

Generates a Linux-kernel-style "Boot Log" for the Pluribus AGI-OS.
It inspects actual system state (Rhizome, Bus, Registry, InferCells)
and translates it into the high-level Pluribus Idiolect/Lexicon.

Usage:
    python3 slou_motd.py --emit-bus
    python3 slou_motd.py --stream (Simulates boot delay)
"""
import argparse
import json
import os
import sys
import time
import random
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

TOOLS_DIR = Path(__file__).resolve().parent
# Import Pluribus internals for grounding
sys.path.append(str(Path(__file__).resolve().parents[2]))
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))
try:
    from nucleus.tools.service_registry import ServiceRegistry
    from nucleus.tools.infercell_manager import InferCellManager
except ImportError:
    pass
try:
    import pulse_prompt
except Exception:
    pulse_prompt = None

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

class SlouGenerator:
    def __init__(self, root: Path):
        self.root = root
        self.t0 = time.time()
        self.registry = ServiceRegistry(root)
        self.registry.load()
        self.infer_mgr = InferCellManager(root)
        self.bus_path = self.root / ".pluribus/bus/events.ndjson"
        self.vps_session_path = self.root / ".pluribus/vps_session.json"
        self.browser_daemon_path = self.root / ".pluribus/browser_daemon.json"
        self.art_history_path = self.root / "nucleus/art_dept/artifacts/history.ndjson"

    def _log(self, subsystem: str, message: str, status: str = "OK") -> dict:
        dt = time.time() - self.t0
        # Kernel-style timestamp [    0.123456]
        ts_str = f"[{dt:12.6f}]"
        status_suffix = f" \033[92m{status}\033[0m" if status else ""
        return {
            "ts": dt,
            "ts_str": ts_str,
            "subsystem": subsystem,
            "message": message,
            "status": status,
            "formatted": f"\033[90m{ts_str}\033[0m \033[1;34m{subsystem.ljust(12)}\033[0m {message}{status_suffix}"
        }

    def generate_boot_sequence(self) -> List[dict]:
        logs = []
        # Build dense “pulse panels” instead of ASCII art.
        panels = self._build_pulse_panels()
        for line in panels:
            logs.append(self._log("PULSE", line, status=""))
        return logs

    def _build_pulse_panels(self) -> List[str]:
        width = max(60, shutil.get_terminal_size((100, 20)).columns)
        if pulse_prompt and hasattr(pulse_prompt, "build_pulse_lines"):
            return pulse_prompt.build_pulse_lines(width)
        panels = [
            self._bus_panel(),
            self._providers_panel(),
            self._browser_panel(),
            self._services_panel(),
            self._hexis_panel(),
            self._art_panel(),
        ]
        rows = []
        for panel in panels:
            label, value = self._split_panel(panel)
            rows.append(self._format_row(label, value))
        title = f"PLURIBUS PULSE {now_iso()}"
        return self._make_box(title, rows, width)

    def _split_panel(self, panel: str) -> Tuple[str, str]:
        if panel.startswith("[") and "]" in panel:
            end = panel.find("]")
            label = panel[1:end]
            value = panel[end + 1 :].strip()
            return label, value
        return "INFO", panel

    def _format_row(self, label: str, value: str) -> str:
        return f"{label.upper():<9} {value}"

    def _clamp(self, text: str, max_len: int) -> str:
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."

    def _make_box(self, title: str, rows: List[str], width: int) -> List[str]:
        width = max(60, width)
        top = "+" + "-" * (width - 2) + "+"

        def box_line(content: str) -> str:
            content = self._clamp(content, width - 4)
            padding = " " * max(0, (width - 4) - len(content))
            return f"| {content}{padding} |"

        lines = [top, box_line(title)]
        lines.extend(box_line(row) for row in rows)
        lines.append(top)
        return lines

    def _bus_panel(self) -> str:
        size_mb = "n/a"
        last_iso = "n/a"
        if self.bus_path.exists():
            try:
                size_mb = f"{self.bus_path.stat().st_size/1024/1024:.2f}MB"
                with self.bus_path.open("rb") as f:
                    f.seek(max(self.bus_path.stat().st_size - 4096, 0))
                    tail = f.read().decode(errors="ignore").strip().splitlines()
                    for line in reversed(tail):
                        try:
                            last_iso = json.loads(line).get("iso") or "n/a"
                            break
                        except Exception:
                            continue
            except Exception:
                pass
        return f"[BUS] size={size_mb} last={last_iso}"

    def _providers_panel(self) -> str:
        if not self.vps_session_path.exists():
            return "[PROVIDERS] vps_session.json missing"
        try:
            data = json.loads(self.vps_session_path.read_text())
            providers = data.get("providers", {})
            def fmt(name):
                entry = providers.get(name, {})
                err = entry.get("error") or ""
                avail = entry.get("available")
                status = "OK" if avail else "blocked"
                if err:
                    status += f" ({err})"
                return f"{name}={status}"
            fallback = data.get("active_fallback") or "none"
            return "[PROVIDERS] " + " | ".join([fmt("gemini-web"), fmt("claude-web"), fmt("chatgpt-web")]) + f" | fallback={fallback}"
        except Exception:
            return "[PROVIDERS] unreadable vps_session.json"

    def _browser_panel(self) -> str:
        if not self.browser_daemon_path.exists():
            return "[BROWSER] daemon state missing"
        try:
            data = json.loads(self.browser_daemon_path.read_text())
            tabs = data.get("tabs", {})
            def fmt(name):
                tab = tabs.get(name, {})
                return f"{name}:{tab.get('status','n/a')}"
            return "[BROWSER] " + " | ".join([fmt("gemini-web"), fmt("claude-web"), fmt("chatgpt-web")])
        except Exception:
            return "[BROWSER] unreadable"

    def _services_panel(self) -> str:
        svc_count = len(self.registry.list_services())
        active_count = len([i for i in self.registry.list_instances() if i.status == 'running'])
        return f"[SERVICES] defs={svc_count} running={active_count}"

    def _hexis_panel(self) -> str:
        counts: Dict[str,int] = {}
        for buf in Path("/tmp").glob("*.buffer"):
            try:
                with buf.open() as f:
                    lines = [l for l in f if l.strip()]
                counts[buf.stem] = len(lines)
            except Exception:
                continue
        if not counts:
            return "[HEXIS] no buffers"
        summary = " ".join(f"{k}:{v}" for k,v in counts.items())
        return f"[HEXIS] {summary}"

    def _art_panel(self) -> str:
        if not self.art_history_path.exists():
            return "[ART] history missing"
        try:
            with self.art_history_path.open() as f:
                tail = f.readlines()[-1:]
            if not tail:
                return "[ART] empty"
            evt = json.loads(tail[0])
            return f"[ART] {evt.get('scene_name','?')} mood={evt.get('mood')} ts={evt.get('ts')}"
        except Exception:
            return "[ART] unreadable"

def main():
    parser = argparse.ArgumentParser(description="SLOU: System Load Ontological Unit (SuperMOTD)")
    parser.add_argument("--root", default="/pluribus")
    parser.add_argument("--stream", action="store_true", help="Simulate boot delay with streaming output")
    parser.add_argument("--emit-bus", action="store_true", help="Publish boot log to bus")
    parser.add_argument("--plain", action="store_true", help="Render pulse panel without kernel-style prefixes")
    args = parser.parse_args()

    gen = SlouGenerator(Path(args.root))

    if os.environ.get("SLOU_PLAIN") == "1":
        args.plain = True

    if args.plain:
        panels = gen._build_pulse_panels()
        for line in panels:
            if args.stream:
                time.sleep(random.uniform(0.01, 0.15))
            print(line)
        if args.emit_bus:
            tool = Path(__file__).with_name("agent_bus.py")
            payload = {"boot_log": panels, "iso": now_iso()}
            subprocess.run([
                sys.executable, str(tool), "pub",
                "--topic", "system.boot.log",
                "--kind", "log",
                "--level", "info",
                "--actor", "slou-daemon",
                "--data", json.dumps(payload)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    sequence = gen.generate_boot_sequence()

    # Output
    for entry in sequence:
        if args.stream:
            time.sleep(random.uniform(0.01, 0.15))
        print(entry["formatted"])

    # Bus Emission
    if args.emit_bus:
        tool = Path(__file__).with_name("agent_bus.py")
        payload = {"boot_log": [e["formatted"] for e in sequence], "iso": now_iso()}
        subprocess.run([
            sys.executable, str(tool), "pub", 
            "--topic", "system.boot.log", 
            "--kind", "log", 
            "--level", "info", 
            "--actor", "slou-daemon", 
            "--data", json.dumps(payload)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()
