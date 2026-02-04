#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools import agent_bus
from nucleus.tools import task_ledger


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iso_git_path() -> Path:
    return Path(__file__).resolve().parent / "iso_git.mjs"


def tail_lines(path: Path, limit: int) -> list[str]:
    if limit <= 0 or not path.exists():
        return []
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line:
                lines.append(line.rstrip("\n"))
    return lines[-limit:]


def summarize_status(status_lines: list[str]) -> dict:
    summary = {
        "untracked": [],
        "modified": [],
        "added": [],
        "deleted": [],
        "renamed": [],
    }
    for line in status_lines:
        if len(line) < 3:
            continue
        code = line[:2]
        path = line[2:].strip()
        if code == "??":
            summary["untracked"].append(path)
        if "M" in code:
            summary["modified"].append(path)
        if "A" in code:
            summary["added"].append(path)
        if "D" in code:
            summary["deleted"].append(path)
        if "R" in code:
            summary["renamed"].append(path)
    return {
        "untracked_count": len(summary["untracked"]),
        "modified_count": len(summary["modified"]),
        "added_count": len(summary["added"]),
        "deleted_count": len(summary["deleted"]),
        "renamed_count": len(summary["renamed"]),
        "untracked": summary["untracked"],
        "modified": summary["modified"],
        "added": summary["added"],
        "deleted": summary["deleted"],
        "renamed": summary["renamed"],
    }


def build_snapshot(
    *,
    status_lines: list[str],
    log_data: dict | None,
    bus_lines: list[str],
    ledger_lines: list[str],
    actor: str,
    cwd: str,
    run_id: str | None,
    created_iso: str,
) -> dict:
    summary = summarize_status(status_lines)
    return {
        "id": str(uuid.uuid4()),
        "created_iso": created_iso,
        "created_ts": time.time(),
        "actor": actor,
        "cwd": cwd,
        "run_id": run_id,
        "summary": summary,
        "status_lines": status_lines,
        "log": log_data,
        "bus_tail": bus_lines,
        "ledger_tail": ledger_lines,
    }


def run_iso_git(args: list[str], repo_dir: Path) -> subprocess.CompletedProcess:
    if not args:
        raise ValueError("iso_git args must include a command")
    cmd = ["node", str(iso_git_path()), args[0], str(repo_dir)] + args[1:]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def collect_status(repo_dir: Path) -> list[str]:
    result = run_iso_git(["status"], repo_dir)
    if result.returncode != 0:
        return [f"!! iso_git status failed: {result.stderr.strip()}"]
    return [line.rstrip("\n") for line in result.stdout.splitlines() if line.strip()]


def collect_log(repo_dir: Path, limit: int) -> dict | None:
    result = run_iso_git(["log", "--limit", str(limit)], repo_dir)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def collect_snapshot(
    *,
    repo_dir: Path,
    bus_limit: int,
    ledger_limit: int,
    log_limit: int,
    actor: str,
    run_id: str | None,
    bus_dir: str | None,
) -> dict:
    created_iso = now_iso_utc()
    status_lines = collect_status(repo_dir)
    log_data = collect_log(repo_dir, limit=log_limit)

    bus_paths = agent_bus.resolve_bus_paths(bus_dir)
    bus_lines = tail_lines(Path(bus_paths.events_path), bus_limit)

    ledger_path = task_ledger.default_ledger_path()
    ledger_lines = tail_lines(ledger_path, ledger_limit)

    return build_snapshot(
        status_lines=status_lines,
        log_data=log_data,
        bus_lines=bus_lines,
        ledger_lines=ledger_lines,
        actor=actor,
        cwd=str(repo_dir),
        run_id=run_id,
        created_iso=created_iso,
    )


def snapshot_dir() -> Path:
    return task_ledger.resolve_state_dir(for_write=True) / "recovery_snapshots"


def write_snapshot(snapshot: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = snapshot.get("created_iso", now_iso_utc()).replace(":", "").replace("-", "")
    path = directory / f"recovery_snapshot_{timestamp}.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="recovery_snapshot.py", description="Write recovery snapshot")
    ap.add_argument("--repo-dir", default=str(repo_root()))
    ap.add_argument("--bus-limit", type=int, default=200)
    ap.add_argument("--ledger-limit", type=int, default=200)
    ap.add_argument("--log-limit", type=int, default=20)
    ap.add_argument("--actor", default=agent_bus.default_actor())
    ap.add_argument("--run-id")
    ap.add_argument("--bus-dir")
    ap.add_argument("--no-bus", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    snapshot = collect_snapshot(
        repo_dir=repo_dir,
        bus_limit=args.bus_limit,
        ledger_limit=args.ledger_limit,
        log_limit=args.log_limit,
        actor=args.actor,
        run_id=args.run_id,
        bus_dir=args.bus_dir,
    )
    path = write_snapshot(snapshot, snapshot_dir())

    if not args.no_bus:
        paths = agent_bus.resolve_bus_paths(args.bus_dir)
        agent_bus.emit_event(
            paths,
            topic="recovery.snapshot.created",
            kind="artifact",
            level="info",
            actor=args.actor,
            data={
                "path": str(path),
                "summary": snapshot.get("summary"),
                "run_id": args.run_id,
            },
            trace_id=None,
            run_id=args.run_id,
            durable=False,
        )

    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
