#!/usr/bin/env python3
"""
Bus Mirror Daemon

Problem:
  Some agent runtimes (e.g. sandboxed Codex) cannot append to the canonical bus at:
    /pluribus/.pluribus/bus/events.ndjson
  so they fall back to the workspace bus:
    /pluribus/.pluribus_local/bus/events.ndjson

This daemon continuously mirrors new lines from the fallback bus into the canonical
bus so that "tailing the canonical bus" remains the single source of truth for
operators and other agents.

Design goals:
  - append-only: never rewrites either bus
  - idempotent-ish: avoids obvious duplicates by checking recent ids in the dest
  - safe: uses file locks on writes; maintains a small offset state file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class MirrorState:
    offset: int = 0
    src_inode: int | None = None


def _load_state(path: Path) -> MirrorState:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return MirrorState(offset=int(raw.get("offset", 0)), src_inode=raw.get("src_inode"))
    except Exception:
        return MirrorState()


def _save_state(path: Path, state: MirrorState, *, src_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "offset": state.offset,
        "src_inode": state.src_inode,
        "src_path": str(src_path),
        "updated_iso": now_iso_utc(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_recent_ids(dest_events: Path, *, bytes_back: int, max_ids: int) -> set[str]:
    if not dest_events.exists():
        return set()
    try:
        size = dest_events.stat().st_size
        start = max(0, size - max(0, bytes_back))
        with dest_events.open("rb") as f:
            f.seek(start, os.SEEK_SET)
            chunk = f.read()
        text = chunk.decode("utf-8", errors="replace")
        if start > 0:
            # Drop any partial first line.
            nl = text.find("\n")
            if nl >= 0:
                text = text[nl + 1 :]
            else:
                text = ""
        ids: list[str] = []
        for line in text.splitlines()[-max_ids:]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            event_id = obj.get("id")
            if isinstance(event_id, str) and event_id:
                ids.append(event_id)
        return set(ids)
    except Exception:
        return set()


def _append_lines(dest_events: Path, lines: list[bytes]) -> None:
    dest_events.parent.mkdir(parents=True, exist_ok=True)
    with dest_events.open("ab") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            for line in lines:
                f.write(line)
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)


def mirror_once(
    *,
    src_events: Path,
    dest_events: Path,
    state: MirrorState,
    recent_bytes_back: int,
    max_recent_ids: int,
) -> tuple[MirrorState, dict]:
    """Mirror any new complete lines from src into dest, updating `state`."""
    src_events.parent.mkdir(parents=True, exist_ok=True)
    src_events.touch(exist_ok=True)

    st = src_events.stat()
    src_inode = getattr(st, "st_ino", None)
    src_size = st.st_size
    offset = max(0, int(state.offset or 0))

    # Rotation / truncation safety: reset if file shrank or inode changed.
    if state.src_inode is not None and src_inode is not None and state.src_inode != src_inode:
        offset = 0
    if src_size < offset:
        offset = 0

    state.src_inode = src_inode
    state.offset = offset

    if src_size == offset:
        return state, {"mirrored": 0, "skipped": 0, "src_size": src_size, "offset": offset}

    recent_ids = _read_recent_ids(dest_events, bytes_back=recent_bytes_back, max_ids=max_recent_ids)
    to_write: list[bytes] = []
    mirrored = 0
    skipped = 0

    with src_events.open("rb") as f:
        f.seek(offset, os.SEEK_SET)
        while True:
            line = f.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                # Defensive: don't advance offset past a partial line.
                f.seek(-len(line), os.SEEK_CUR)
                break

            event_id: str | None = None
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
                if isinstance(obj, dict):
                    eid = obj.get("id")
                    if isinstance(eid, str) and eid:
                        event_id = eid
            except Exception:
                event_id = None

            if event_id and event_id in recent_ids:
                skipped += 1
                continue

            to_write.append(line)
            mirrored += 1
            if event_id:
                recent_ids.add(event_id)

        state.offset = f.tell()

    if to_write:
        _append_lines(dest_events, to_write)

    return state, {"mirrored": mirrored, "skipped": skipped, "src_size": src_size, "offset": state.offset}


def emit_bus_metric(*, bus_dir: Path, topic: str, actor: str, data: dict) -> None:
    # Local import keeps this tool usable as a standalone script.
    try:
        tools_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(tools_dir))
        import agent_bus  # type: ignore

        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind="metric",
            level="info",
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
    except Exception:
        return


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Mirror fallback bus events into canonical bus.")
    ap.add_argument("--from-bus-dir", default="/pluribus/.pluribus_local/bus", help="Source bus dir (fallback).")
    ap.add_argument("--to-bus-dir", default="/pluribus/.pluribus/bus", help="Destination bus dir (canonical).")
    ap.add_argument("--state-path", default=None, help="Offset state file (default: <to>/bus_mirror_state.json).")
    ap.add_argument("--poll", type=float, default=0.5, help="Polling interval seconds.")
    ap.add_argument("--once", action="store_true", help="Mirror current delta once and exit.")
    ap.add_argument(
        "--start-at-end",
        action="store_true",
        help="On first run (no state file yet), start mirroring from EOF to avoid copying historical backlog.",
    )
    ap.add_argument("--recent-bytes-back", type=int, default=2_000_000, help="Bytes to scan from dest tail for dedup.")
    ap.add_argument("--max-recent-ids", type=int, default=20_000, help="Max recent ids to consider for dedup.")
    ap.add_argument("--emit-bus", action="store_true", help="Emit bus.mirror.batch metrics when mirroring.")
    ap.add_argument(
        "--emit-interval-s",
        type=float,
        default=10.0,
        help="Minimum seconds between bus.mirror.batch emissions (0 = every loop).",
    )
    ap.add_argument("--actor", default="bus-mirror", help="Actor name for metric emission.")
    args = ap.parse_args(argv)

    src_bus = Path(args.from_bus_dir).expanduser().resolve()
    dest_bus = Path(args.to_bus_dir).expanduser().resolve()
    src_events = src_bus / "events.ndjson"
    dest_events = dest_bus / "events.ndjson"
    state_path = Path(args.state_path).expanduser().resolve() if args.state_path else (dest_bus / "bus_mirror_state.json")
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_f:
        if fcntl is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                sys.stderr.write("bus_mirror_daemon: already running (lock busy)\n")
                return 2

        first_run = not state_path.exists()
        state = _load_state(state_path)
        if args.start_at_end and first_run:
            try:
                st = src_events.stat()
                state.offset = st.st_size
                state.src_inode = getattr(st, "st_ino", None)
                _save_state(state_path, state, src_path=src_events)
            except Exception:
                # If we can't stat the source, we just proceed from offset=0.
                pass
        last_emit_ts = 0.0
        while True:
            state, stats = mirror_once(
                src_events=src_events,
                dest_events=dest_events,
                state=state,
                recent_bytes_back=args.recent_bytes_back,
                max_recent_ids=args.max_recent_ids,
            )
            _save_state(state_path, state, src_path=src_events)

            mirrored = int(stats.get("mirrored") or 0)
            should_emit = args.emit_bus and mirrored > 0
            if should_emit:
                interval = float(args.emit_interval_s or 0.0)
                now = time.time()
                if interval > 0 and (now - last_emit_ts) < interval:
                    should_emit = False
                else:
                    last_emit_ts = now

            if should_emit:
                emit_bus_metric(
                    bus_dir=dest_bus,
                    topic="bus.mirror.batch",
                    actor=args.actor,
                    data={
                        **stats,
                        "from_bus_dir": str(src_bus),
                        "to_bus_dir": str(dest_bus),
                        "state_path": str(state_path),
                        "iso": now_iso_utc(),
                    },
                )

            if args.once:
                return 0
            time.sleep(max(0.05, float(args.poll)))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
