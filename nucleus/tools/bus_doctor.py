#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

DEFAULT_MAX_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_AGE_HOURS = 24.0
DEFAULT_TAIL_LINES = 1000
DEFAULT_BLOCK_SIZE = 64 * 1024


def now_ts() -> float:
    return time.time()


def iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso_ts(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def normalize_ts(value: float) -> float:
    if value > 1e12:
        return value / 1000.0
    return value


def resolve_root(env: dict[str, str]) -> Path:
    root = (env.get("PLURIBUS_ROOT") or "/pluribus").strip() or "/pluribus"
    return Path(root)


def resolve_bus_dir(env: dict[str, str], override: str | None) -> Path:
    if override:
        return Path(override)
    if env.get("PLURIBUS_BUS_DIR"):
        return Path(env["PLURIBUS_BUS_DIR"])
    return resolve_root(env) / ".pluribus" / "bus"


def resolve_archive_dir(env: dict[str, str], override: str | None, *, now: datetime) -> Path:
    if override:
        return Path(override)
    if env.get("PLURIBUS_LOG_ARCHIVE_DIR"):
        return Path(env["PLURIBUS_LOG_ARCHIVE_DIR"])
    root = resolve_root(env)
    return root / "agent_logs" / "archive" / now.strftime("%Y-%m")


def tail_lines(path: Path, *, max_lines: int, block_size: int = DEFAULT_BLOCK_SIZE) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    data = b""
    size = path.stat().st_size
    with path.open("rb") as fh:
        pos = size
        while pos > 0 and data.count(b"\n") <= max_lines:
            step = min(block_size, pos)
            pos -= step
            fh.seek(pos)
            data = fh.read(step) + data
    lines = data.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return [line.decode("utf-8", errors="replace") for line in lines]


def extract_event_ts(obj: dict) -> tuple[float | None, str | None]:
    ts_val = obj.get("ts")
    iso_val = obj.get("iso")
    ts = None
    if isinstance(ts_val, (int, float)):
        ts = normalize_ts(float(ts_val))
    elif isinstance(ts_val, str):
        try:
            ts = normalize_ts(float(ts_val))
        except Exception:
            ts = None
    if ts is None and isinstance(iso_val, str):
        ts = parse_iso_ts(iso_val)
    iso = iso_val if isinstance(iso_val, str) and iso_val.strip() else None
    if ts is not None and not iso:
        iso = iso_from_ts(ts)
    return ts, iso


@dataclass
class BusReport:
    bus_dir: str
    events_path: str
    size_bytes: int
    last_event_ts: float | None
    last_event_iso: str | None
    age_seconds: float | None
    tail_lines: int
    json_valid: int
    json_errors: int
    needs_rotation: bool
    rotation_reasons: list[str]
    rotated: bool = False
    archive_path: str | None = None
    note_path: str | None = None

    def to_json(self) -> dict:
        return {
            "bus_dir": self.bus_dir,
            "events_path": self.events_path,
            "size_bytes": self.size_bytes,
            "last_event_ts": self.last_event_ts,
            "last_event_iso": self.last_event_iso,
            "age_seconds": self.age_seconds,
            "tail_lines": self.tail_lines,
            "json_valid": self.json_valid,
            "json_errors": self.json_errors,
            "needs_rotation": self.needs_rotation,
            "rotation_reasons": list(self.rotation_reasons),
            "rotated": self.rotated,
            "archive_path": self.archive_path,
            "note_path": self.note_path,
        }


def inspect_tail(lines: list[str]) -> tuple[int, int, float | None, str | None]:
    json_valid = 0
    json_errors = 0
    last_ts = None
    last_iso = None
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            json_errors += 1
            continue
        if not isinstance(obj, dict):
            json_errors += 1
            continue
        json_valid += 1
        ts, iso = extract_event_ts(obj)
        if ts is not None:
            last_ts = ts
            last_iso = iso
    return json_valid, json_errors, last_ts, last_iso


def inspect_bus(
    *,
    bus_dir: Path,
    events_path: Path | None,
    max_bytes: int,
    max_age_hours: float | None,
    tail_max_lines: int,
    now: float | None = None,
) -> BusReport:
    bus_dir = bus_dir.resolve()
    events = events_path.resolve() if events_path else bus_dir / "events.ndjson"
    size_bytes = events.stat().st_size if events.exists() else 0
    lines = tail_lines(events, max_lines=tail_max_lines)
    json_valid, json_errors, last_ts, last_iso = inspect_tail(lines)
    now_ts_val = now if now is not None else now_ts()
    age_seconds = (now_ts_val - last_ts) if last_ts is not None else None

    reasons: list[str] = []
    if max_bytes > 0 and size_bytes >= max_bytes:
        reasons.append("size")
    if max_age_hours and max_age_hours > 0 and age_seconds is not None:
        if age_seconds >= max_age_hours * 3600:
            reasons.append("age")
    if json_errors > 0:
        reasons.append("json_errors")

    return BusReport(
        bus_dir=str(bus_dir),
        events_path=str(events),
        size_bytes=size_bytes,
        last_event_ts=last_ts,
        last_event_iso=last_iso,
        age_seconds=age_seconds,
        tail_lines=tail_max_lines,
        json_valid=json_valid,
        json_errors=json_errors,
        needs_rotation=bool(reasons),
        rotation_reasons=reasons,
    )


def ensure_events_file(path: Path, *, mode: int = 0o666) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    try:
        os.chmod(path, mode)
    except Exception:
        pass


def build_archive_note(
    *,
    events_path: Path,
    archive_path: Path,
    size_bytes: int,
    rotated_iso: str,
    reasons: list[str],
    last_event_iso: str | None,
) -> str:
    reasons_text = ", ".join(reasons) if reasons else "manual"
    lines = [
        f"Archived from {events_path} on {rotated_iso}.",
        f"Size: {size_bytes} bytes.",
        f"Reasons: {reasons_text}.",
    ]
    if last_event_iso:
        lines.append(f"Last event ISO: {last_event_iso}.")
    lines.append(f"Archive path: {archive_path}.")
    return "\n".join(lines) + "\n"


def rotate_events(
    report: BusReport,
    *,
    archive_dir: Path,
    dry_run: bool,
) -> tuple[str | None, str | None]:
    events_path = Path(report.events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    if not events_path.exists():
        if not dry_run:
            ensure_events_file(events_path)
        return None, None

    stat = events_path.stat()
    if stat.st_size == 0:
        if not dry_run:
            ensure_events_file(events_path, mode=stat.st_mode & 0o777 or 0o666)
        return None, None

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{events_path.name}.{stamp}"
    note_path = Path(f"{archive_path}.note.txt")

    if dry_run:
        return str(archive_path), str(note_path)

    try:
        events_path.rename(archive_path)
    except OSError:
        shutil.move(str(events_path), str(archive_path))

    ensure_events_file(events_path, mode=stat.st_mode & 0o777 or 0o666)

    note = build_archive_note(
        events_path=events_path,
        archive_path=archive_path,
        size_bytes=stat.st_size,
        rotated_iso=now.isoformat().replace("+00:00", "Z"),
        reasons=report.rotation_reasons,
        last_event_iso=report.last_event_iso,
    )
    note_path.write_text(note, encoding="utf-8")
    return str(archive_path), str(note_path)


def format_human(report: BusReport) -> str:
    parts = [
        f"bus_dir: {report.bus_dir}",
        f"events: {report.events_path}",
        f"size_bytes: {report.size_bytes}",
        f"last_event_iso: {report.last_event_iso or 'unknown'}",
        f"age_seconds: {report.age_seconds if report.age_seconds is not None else 'unknown'}",
        f"tail_lines: {report.tail_lines}",
        f"json_valid: {report.json_valid}",
        f"json_errors: {report.json_errors}",
        f"needs_rotation: {report.needs_rotation}",
        f"rotation_reasons: {', '.join(report.rotation_reasons) if report.rotation_reasons else 'none'}",
    ]
    if report.rotated:
        parts.append(f"rotated: {report.rotated}")
        if report.archive_path:
            parts.append(f"archive_path: {report.archive_path}")
        if report.note_path:
            parts.append(f"note_path: {report.note_path}")
    return "\n".join(parts)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Bus doctor: inspect/rotate events.ndjson health.")
    ap.add_argument("--bus-dir", default=None, help="Bus directory (default: PLURIBUS_BUS_DIR or /pluribus/.pluribus/bus).")
    ap.add_argument("--events-path", default=None, help="Override events.ndjson path.")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Rotate if size exceeds this many bytes.")
    ap.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS, help="Rotate if last event older than this.")
    ap.add_argument("--tail-lines", type=int, default=DEFAULT_TAIL_LINES, help="Lines to validate from tail.")
    ap.add_argument("--rotate", action="store_true", help="Rotate if thresholds are exceeded.")
    ap.add_argument("--force", action="store_true", help="Force rotation even if thresholds are not exceeded.")
    ap.add_argument("--archive-dir", default=None, help="Archive directory for rotated logs.")
    ap.add_argument("--dry-run", action="store_true", help="Report actions without writing changes.")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output.")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    bus_dir = resolve_bus_dir(env, args.bus_dir)
    events_path = Path(args.events_path) if args.events_path else None
    max_age = args.max_age_hours if args.max_age_hours and args.max_age_hours > 0 else None

    report = inspect_bus(
        bus_dir=bus_dir,
        events_path=events_path,
        max_bytes=args.max_bytes,
        max_age_hours=max_age,
        tail_max_lines=args.tail_lines,
    )

    should_rotate = args.force or (args.rotate and report.needs_rotation)
    if should_rotate:
        archive_dir = resolve_archive_dir(env, args.archive_dir, now=datetime.now(timezone.utc))
        archive_path, note_path = rotate_events(report, archive_dir=archive_dir, dry_run=args.dry_run)
        report.rotated = True
        report.archive_path = archive_path
        report.note_path = note_path
        if args.force and "forced" not in report.rotation_reasons:
            report.rotation_reasons.append("forced")

    if args.json:
        print(json.dumps(report.to_json(), ensure_ascii=False))
    else:
        print(format_human(report))

    if report.needs_rotation and not should_rotate:
        return 1
    if report.json_errors > 0 and not report.rotated:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
