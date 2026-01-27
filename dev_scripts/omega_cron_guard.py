#!/usr/bin/env python3
"""
omega_cron_guard.py - lightweight cron-safe process guard

Goal:
- Detect long-running, duplicate, or orphaned Pluribus tool calls (python/node)
- Report estimated CPU/RSS savings
- Optionally terminate duplicates/orphans

Default: report-only (no kills).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ProcRow:
    pid: int
    ppid: int
    etimes: int
    pcpu: float
    pmem: float
    rss_kb: int
    stat: str
    comm: str
    args: str


ALLOW_PATTERNS = [
    r"/pluribus/nucleus/tools/(omega|dialogos|falkordb|bus_mirror|world_router)",
    r"/pluribus/nucleus/tools/.*daemon\.py",
    r"/pluribus/nucleus/dashboard",
    r"/pluribus/nucleus/tools/(recruit|metaingest|portal|vnc|browser)",
    r"sshd|systemd|redis|postgres|Xvfb|chrom|playwright",
    r"/root/.local/node20/bin/codex",
]

SINGLETON_PATTERNS = [
    r"/pluribus/nucleus/tools/log_hygiene\.py",
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_processes() -> List[ProcRow]:
    raw = subprocess.check_output(
        ["ps", "-eo", "pid,ppid,etimes,pcpu,pmem,rss,stat,comm,args", "--sort=-etimes"],
        text=True,
    )
    rows: List[ProcRow] = []
    for line in raw.strip().splitlines()[1:]:
        parts = line.strip().split(None, 8)
        if len(parts) < 9:
            continue
        pid, ppid, etimes, pcpu, pmem, rss, stat, comm, args = parts
        rows.append(
            ProcRow(
                pid=int(pid),
                ppid=int(ppid),
                etimes=int(etimes),
                pcpu=float(pcpu),
                pmem=float(pmem),
                rss_kb=int(rss),
                stat=stat,
                comm=comm,
                args=args,
            )
        )
    return rows


def _matches(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _is_tool_process(row: ProcRow) -> bool:
    if row.comm not in {"python", "python3", "node"}:
        return False
    if "/pluribus" not in row.args:
        return False
    return True


def _group_duplicates(rows: List[ProcRow]) -> List[ProcRow]:
    dupes: List[ProcRow] = []
    for pattern in SINGLETON_PATTERNS:
        matches = [r for r in rows if re.search(pattern, r.args)]
        if len(matches) <= 1:
            continue
        # Keep the newest (smallest etimes); flag the rest
        matches.sort(key=lambda r: r.etimes)
        dupes.extend(matches[1:])
    return dupes


def _find_orphans(rows: List[ProcRow], min_age: int) -> List[ProcRow]:
    orphans: List[ProcRow] = []
    for r in rows:
        if r.etimes < min_age:
            continue
        if r.ppid != 1:
            continue
        if not _is_tool_process(r):
            continue
        if _matches(ALLOW_PATTERNS, r.args):
            continue
        orphans.append(r)
    return orphans


def _find_zombies(rows: List[ProcRow], min_age: int) -> List[ProcRow]:
    zombies: List[ProcRow] = []
    for r in rows:
        if r.etimes < min_age:
            continue
        if "Z" not in r.stat:
            continue
        if not _is_tool_process(r):
            continue
        if _matches(ALLOW_PATTERNS, r.args):
            continue
        zombies.append(r)
    return zombies


def _signature(row: ProcRow) -> str:
    return f"{row.comm}|{row.args}".strip()


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def _estimate_savings(rows: List[ProcRow]) -> dict:
    total_rss_kb = sum(r.rss_kb for r in rows)
    total_cpu = sum(r.pcpu for r in rows)
    return {
        "rss_mb": round(total_rss_kb / 1024, 2),
        "cpu_pct": round(total_cpu, 2),
        "count": len(rows),
    }


def _log_event(path: Path, payload: dict, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if fmt == "jsonl":
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            if payload.get("action") in {"report", "kill"}:
                summary = payload.get("estimated_savings", {})
                line = (
                    f"{payload.get('iso')} action={payload.get('action')} "
                    f"candidates={summary.get('count')} "
                    f"rss_mb={summary.get('rss_mb')} cpu_pct={summary.get('cpu_pct')}"
                )
            else:
                line = (
                    f"{payload.get('iso')} action=result "
                    f"killed={len(payload.get('killed', []))} "
                    f"failed={len(payload.get('failed', []))}"
                )
            handle.write(line + "\n")


def _terminate(pid: int, grace_s: float) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    deadline = time.time() + grace_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Cron-safe toolcall guard (report/kill).")
    ap.add_argument("--min-age", type=int, default=600, help="Minimum age in seconds (default 600)")
    ap.add_argument("--kill", action="store_true", help="Terminate matched processes")
    ap.add_argument("--grace", type=float, default=2.0, help="SIGTERM grace seconds")
    ap.add_argument("--log", default="/pluribus/dev_scripts/omega_cron_guard.log")
    ap.add_argument("--log-format", choices=["text", "jsonl"], default="text")
    ap.add_argument("--state", default="/pluribus/dev_scripts/omega_cron_guard_state.json")
    ap.add_argument("--cooldown", type=int, default=180, help="Cooldown seconds per command signature")
    args = ap.parse_args()

    rows = _load_processes()
    tool_rows = [r for r in rows if _is_tool_process(r) and r.etimes >= args.min_age]
    tool_rows = [r for r in tool_rows if not _matches(ALLOW_PATTERNS, r.args)]

    dupes = _group_duplicates(tool_rows)
    orphans = _find_orphans(tool_rows, args.min_age)
    zombies = _find_zombies(tool_rows, args.min_age)

    # Merge unique candidates
    candidate_map = {r.pid: r for r in dupes + orphans + zombies}
    candidates = list(candidate_map.values())
    candidates.sort(key=lambda r: r.etimes, reverse=True)

    reason_map: dict[int, list[str]] = {}
    for r in dupes:
        reason_map.setdefault(r.pid, []).append("duplicate_singleton")
    for r in orphans:
        reason_map.setdefault(r.pid, []).append("orphaned")
    for r in zombies:
        reason_map.setdefault(r.pid, []).append("zombie")

    state_path = Path(args.state)
    state = _load_state(state_path)
    now = time.time()

    def cooldown_remaining(sig: str) -> int:
        last = state.get(sig)
        if not isinstance(last, (int, float)):
            return 0
        remaining = int(args.cooldown - (now - float(last)))
        return max(0, remaining)

    savings = _estimate_savings(candidates)
    report = {
        "ts": time.time(),
        "iso": _now_iso(),
        "min_age": args.min_age,
        "candidates": [
            {
                "pid": r.pid,
                "ppid": r.ppid,
                "etimes_s": r.etimes,
                "pcpu": r.pcpu,
                "pmem": r.pmem,
                "rss_kb": r.rss_kb,
                "stat": r.stat,
                "comm": r.comm,
                "args": r.args,
                "reasons": reason_map.get(r.pid, []),
                "cooldown_remaining_s": cooldown_remaining(_signature(r)),
            }
            for r in candidates
        ],
        "estimated_savings": savings,
        "action": "kill" if args.kill else "report",
    }

    print(json.dumps(report, indent=2))
    _log_event(Path(args.log), report, args.log_format)

    if not args.kill or not candidates:
        return 0

    killed = []
    failed = []
    skipped = []
    for r in candidates:
        if "zombie" in reason_map.get(r.pid, []):
            skipped.append(r.pid)
            continue
        sig = _signature(r)
        remaining = cooldown_remaining(sig)
        if remaining > 0:
            skipped.append(r.pid)
            continue
        ok = _terminate(r.pid, args.grace)
        if ok:
            killed.append(r.pid)
            state[sig] = now
        else:
            failed.append(r.pid)

    result = {
        "ts": time.time(),
        "iso": _now_iso(),
        "killed": killed,
        "failed": failed,
        "skipped": skipped,
        "count": len(killed),
    }
    print(json.dumps(result, indent=2))
    _log_event(Path(args.log), result, args.log_format)
    _save_state(state_path, state)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
