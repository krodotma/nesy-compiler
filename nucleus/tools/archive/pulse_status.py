#!/usr/bin/env python3
"""
Emit Pulse status snapshot to the bus.
Gathers: bus size/last iso, providers, browser tabs, services running count, HEXIS pending, art latest.
"""
from __future__ import annotations
import json
import os
import time
import fcntl
from pathlib import Path
import argparse

BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
BUS_EVENTS = BUS_DIR / "events.ndjson"

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def bus_snapshot():
    size_mb = None
    last_iso = None
    if BUS_EVENTS.exists():
        try:
            size_mb = round(BUS_EVENTS.stat().st_size / 1024 / 1024, 2)
            with BUS_EVENTS.open("rb") as f:
                f.seek(max(BUS_EVENTS.stat().st_size - 4096, 0))
                tail = f.read().decode(errors="ignore").strip().splitlines()
                for line in reversed(tail):
                    try:
                        last_iso = json.loads(line).get("iso") or None
                        break
                    except Exception:
                        continue
        except Exception:
            pass
    return {"size_mb": size_mb, "last_iso": last_iso}

def providers_snapshot(vps_path: Path):
    data = read_json(vps_path, {})
    providers = data.get("providers", {})
    out = {}
    for name in ("gemini-web", "claude-web", "chatgpt-web"):
        entry = providers.get(name, {})
        out[name] = {
            "available": entry.get("available"),
            "error": entry.get("error"),
            "model": entry.get("model"),
            "last_check": entry.get("last_check"),
        }
    out["active_fallback"] = data.get("active_fallback")
    return out

def browser_snapshot(browser_path: Path):
    data = read_json(browser_path, {})
    tabs = data.get("tabs", {})
    out = {}
    for name in ("gemini-web", "claude-web", "chatgpt-web"):
        tab = tabs.get(name, {})
        out[name] = {"status": tab.get("status"), "error": tab.get("error")}
    return out

def services_snapshot(registry_path: Path):
    try:
        from nucleus.tools.service_registry import ServiceRegistry
    except Exception:
        return {}
    reg = ServiceRegistry(Path("/pluribus"))
    reg.load()
    return {
        "definitions": len(reg.list_services()),
        "running": len([i for i in reg.list_instances() if i.status == "running"]),
    }

def hexis_snapshot():
    counts = {}
    for buf in Path("/tmp").glob("*.buffer"):
        try:
            with buf.open() as f:
                lines = [l for l in f if l.strip()]
            counts[buf.stem] = len(lines)
        except Exception:
            continue
    return counts

def art_snapshot(path: Path):
    if not path.exists():
        return None
    try:
        with path.open() as f:
            tail = f.readlines()[-1:]
        if not tail:
            return None
        evt = json.loads(tail[0])
        return {
            "scene": evt.get("scene_name"),
            "mood": evt.get("mood"),
            "ts": evt.get("ts"),
        }
    except Exception:
        return None

def emit(topic: str, actor: str, data: dict, bus_dir: Path):
    event = {
        "id": os.urandom(16).hex(),
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": "metric",
        "level": "info",
        "actor": actor,
        "data": data,
    }
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bus-dir", default=str(BUS_DIR))
    parser.add_argument("--actor", default="pulse")
    args = parser.parse_args()

    bus_dir = Path(args.bus_dir)
    data = {
        "bus": bus_snapshot(),
        "providers": providers_snapshot(Path("/var/lib/pluribus/.pluribus/vps_session.json")),
        "browser": browser_snapshot(Path("/var/lib/pluribus/.pluribus/browser_daemon.json")),
        "services": services_snapshot(Path("/pluribus/.pluribus/services/registry.json")),
        "hexis": hexis_snapshot(),
        "art": art_snapshot(Path("/pluribus/nucleus/art_dept/artifacts/history.ndjson")),
    }
    emit("pulse.status", args.actor, data, bus_dir)

if __name__ == "__main__":
    main()
