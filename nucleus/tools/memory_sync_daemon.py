#!/usr/bin/env python3
"""
memory_sync_daemon.py - lightweight memory sync daemon.

Purpose:
- Periodically sync shared memories with peer agents using agent_memory_sync.py.
- Optionally emit bus events with sync stats.

This is a conservative implementation intended to keep the legacy
pluribus-memory-sync.service functional without requiring additional
infrastructure. It does NOT assume any particular bus event schema.

Environment:
  PLURIBUS_ROOT                     (default: /pluribus)
  PLURIBUS_ACTOR                    (default: memory_sync_daemon)
  PLURIBUS_BUS_DIR                  (default: <root>/.pluribus/bus)
  PLURIBUS_MEMORY_SYNC_PEERS        (comma/space-separated peers)
  PLURIBUS_SYNC_PEERS               (alias)
  PLURIBUS_MEMORY_SYNC_PEERS_FILE   (JSON list or newline-separated peers)
  MEMORY_SYNC_INTERVAL_S            (default: 300)
  MEMORY_SYNC_ON_START              (default: 1)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def _parse_peers(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts = []
    for chunk in raw.replace("\n", ",").replace(" ", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def _load_peers_file(path: Path) -> List[str]:
    try:
        data = path.read_text(encoding="utf-8").strip()
    except Exception:
        return []
    if not data:
        return []
    # Try JSON first
    try:
        parsed = json.loads(data)
        if isinstance(parsed, list):
            return [str(p).strip() for p in parsed if str(p).strip()]
        if isinstance(parsed, dict):
            for key in ("peers", "agents", "participants"):
                val = parsed.get(key)
                if isinstance(val, list):
                    return [str(p).strip() for p in val if str(p).strip()]
    except Exception:
        pass
    # Fallback: newline-separated
    return [line.strip() for line in data.splitlines() if line.strip()]


def _resolve_peers(actor: str) -> List[str]:
    peers: List[str] = []
    peers.extend(_parse_peers(os.environ.get("PLURIBUS_MEMORY_SYNC_PEERS")))
    peers.extend(_parse_peers(os.environ.get("PLURIBUS_SYNC_PEERS")))
    peers_file = os.environ.get("PLURIBUS_MEMORY_SYNC_PEERS_FILE")
    if peers_file:
        peers.extend(_load_peers_file(Path(peers_file)))
    # dedupe + remove self
    uniq = []
    seen = set()
    for peer in peers:
        if peer == actor:
            continue
        if peer in seen:
            continue
        seen.add(peer)
        uniq.append(peer)
    return uniq


def _emit_bus_event(bus_dir: Path, actor: str, topic: str, data: dict, level: str = "info") -> None:
    try:
        tools_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(tools_dir))
        import agent_bus  # type: ignore

        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind="event",
            level=level,
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
    except Exception:
        return


def _run_sync(root: Path, actor: str, peer: str) -> dict:
    cmd = [
        sys.executable,
        str(root / "nucleus" / "tools" / "agent_memory_sync.py"),
        "--root",
        str(root),
        "--agent",
        actor,
        "sync",
        "--peer",
        peer,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    payload: dict = {
        "peer": peer,
        "returncode": result.returncode,
    }
    if result.stdout:
        try:
            payload["result"] = json.loads(result.stdout)
        except Exception:
            payload["stdout"] = result.stdout.strip()
    if result.stderr:
        payload["stderr"] = result.stderr.strip()
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pluribus memory sync daemon")
    parser.add_argument("--once", action="store_true", help="Sync once and exit")
    parser.add_argument("--interval", type=int, default=None, help="Sync interval seconds")
    args = parser.parse_args(argv)

    root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus")).resolve()
    actor = os.environ.get("PLURIBUS_ACTOR", "memory_sync_daemon")
    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", str(root / ".pluribus" / "bus"))).resolve()

    interval = args.interval or int(os.environ.get("MEMORY_SYNC_INTERVAL_S", "300"))
    run_on_start = os.environ.get("MEMORY_SYNC_ON_START", "1") not in ("0", "false", "False")

    if run_on_start:
        peers = _resolve_peers(actor)
        if not peers:
            _emit_bus_event(bus_dir, actor, "memory.sync.idle", {"reason": "no_peers"}, level="warning")
        for peer in peers:
            payload = _run_sync(root, actor, peer)
            _emit_bus_event(bus_dir, actor, "memory.sync.batch", payload, level="info" if payload.get("returncode") == 0 else "warning")
        if args.once:
            return 0

    while True:
        peers = _resolve_peers(actor)
        if not peers:
            _emit_bus_event(bus_dir, actor, "memory.sync.idle", {"reason": "no_peers"}, level="warning")
            time.sleep(max(30, interval))
            if args.once:
                return 0
            continue

        for peer in peers:
            payload = _run_sync(root, actor, peer)
            _emit_bus_event(bus_dir, actor, "memory.sync.batch", payload, level="info" if payload.get("returncode") == 0 else "warning")
        if args.once:
            return 0
        time.sleep(max(10, interval))


if __name__ == "__main__":
    raise SystemExit(main())
