#!/usr/bin/env python3
"""
req_id_registry.py - Request ID collision detection and tracking

Provides a simple file-based registry to track active request IDs
and detect collisions when multiple agents issue the same req_id.

Usage:
    from req_id_registry import RequestIdRegistry

    registry = RequestIdRegistry()

    # Acquire a req_id (returns False if already in use)
    if not registry.acquire(req_id, actor="claude-opus"):
        print(f"Collision! req_id {req_id} already in use")

    # Release when done
    registry.release(req_id)

Registry file: /pluribus/.pluribus/bus/req_id_registry.json
"""
from __future__ import annotations

import json
import os
import time
import fcntl
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict


def _default_registry_path() -> Path:
    """
    Resolve a writable registry path, honoring Pluribus' canonical bus + sandbox fallback.

    We intentionally go through `agent_bus.resolve_bus_paths()` so sandboxed runs that cannot
    write to `/pluribus/.pluribus/bus` still record evidence in `.pluribus_local/bus`.
    """
    try:
        from nucleus.tools.agent_bus import resolve_bus_paths  # type: ignore
    except Exception:  # pragma: no cover
        from agent_bus import resolve_bus_paths  # type: ignore

    try:
        paths = resolve_bus_paths(os.environ.get("PLURIBUS_BUS_DIR"))
        return Path(paths.bus_dir) / "req_id_registry.json"
    except Exception:
        return Path("/pluribus/.pluribus_local/bus") / "req_id_registry.json"


REGISTRY_PATH = _default_registry_path()
STALE_THRESHOLD_S = 300  # 5 minutes - req_ids older than this are considered stale


@dataclass
class RequestIdEntry:
    req_id: str
    actor: str
    acquired_at: float
    topic: str = ""


class RequestIdRegistry:
    """Thread-safe request ID registry with collision detection."""

    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        """Ensure registry file exists."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("{}", encoding="utf-8")

    def _load(self) -> Dict[str, dict]:
        """Load registry from disk."""
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        """Save registry to disk."""
        self.registry_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _prune_stale(self, data: Dict[str, dict]) -> Dict[str, dict]:
        """Remove stale entries older than STALE_THRESHOLD_S."""
        now = time.time()
        return {
            k: v for k, v in data.items()
            if now - v.get("acquired_at", 0) < STALE_THRESHOLD_S
        }

    def acquire(self, req_id: str, actor: str, topic: str = "") -> tuple[bool, Optional[str]]:
        """
        Attempt to acquire a req_id for exclusive use.

        Returns:
            (success, existing_actor) - If collision, existing_actor is who holds it
        """
        try:
            with open(self.registry_path, "r+") as f:
                # File-level locking for thread safety
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    data = json.loads(f.read() or "{}")
                    data = self._prune_stale(data)

                    if req_id in data:
                        existing = data[req_id]
                        return False, existing.get("actor", "unknown")

                    # Register the req_id
                    data[req_id] = asdict(RequestIdEntry(
                        req_id=req_id,
                        actor=actor,
                        acquired_at=time.time(),
                        topic=topic
                    ))

                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(data, indent=2, ensure_ascii=False))
                    return True, None
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            # On error, allow the request (fail-open)
            print(f"[req_id_registry] Warning: {e}")
            return True, None

    def release(self, req_id: str) -> bool:
        """Release a req_id back to the pool."""
        try:
            with open(self.registry_path, "r+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    data = json.loads(f.read() or "{}")
                    if req_id in data:
                        del data[req_id]
                        f.seek(0)
                        f.truncate()
                        f.write(json.dumps(data, indent=2, ensure_ascii=False))
                        return True
                    return False
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            return False

    def check(self, req_id: str) -> Optional[RequestIdEntry]:
        """Check if a req_id is currently acquired."""
        data = self._load()
        data = self._prune_stale(data)
        if req_id in data:
            entry = data[req_id]
            return RequestIdEntry(**entry)
        return None

    def list_active(self) -> list[RequestIdEntry]:
        """List all active (non-stale) req_ids."""
        data = self._load()
        data = self._prune_stale(data)
        return [RequestIdEntry(**v) for v in data.values()]


# CLI interface
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Request ID Registry")
    sub = parser.add_subparsers(dest="cmd")

    acq = sub.add_parser("acquire", help="Acquire a req_id")
    acq.add_argument("req_id")
    acq.add_argument("--actor", default="cli")
    acq.add_argument("--topic", default="")

    rel = sub.add_parser("release", help="Release a req_id")
    rel.add_argument("req_id")

    sub.add_parser("list", help="List active req_ids")

    chk = sub.add_parser("check", help="Check if req_id is in use")
    chk.add_argument("req_id")

    args = parser.parse_args()
    registry = RequestIdRegistry()

    if args.cmd == "acquire":
        ok, existing = registry.acquire(args.req_id, args.actor, args.topic)
        if ok:
            print(f"OK: Acquired {args.req_id}")
        else:
            print(f"COLLISION: {args.req_id} already held by {existing}")
            raise SystemExit(1)

    elif args.cmd == "release":
        if registry.release(args.req_id):
            print(f"OK: Released {args.req_id}")
        else:
            print(f"NOT_FOUND: {args.req_id}")

    elif args.cmd == "list":
        active = registry.list_active()
        if active:
            for e in active:
                print(f"{e.req_id}: {e.actor} ({e.topic}) @ {e.acquired_at}")
        else:
            print("No active req_ids")

    elif args.cmd == "check":
        entry = registry.check(args.req_id)
        if entry:
            print(f"IN_USE: {entry.req_id} by {entry.actor}")
            raise SystemExit(1)
        else:
            print(f"AVAILABLE: {args.req_id}")


if __name__ == "__main__":
    main()
