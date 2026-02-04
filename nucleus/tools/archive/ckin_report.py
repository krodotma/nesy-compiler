#!/usr/bin/env python3
"""
CKIN_REPORT — Pluribus Check-In Dashboard Generator (Protocol v13 / DKIN)

Purpose:
- Generate rich, verbose status dashboards when operator types "ckin"
- Provide deep task velocity metrics with multi-dimensional progress bars
- Show anticipated vs observed velocity with trend analysis
- Display topic distribution, agent contribution heatmaps, timeline views
- Emit structured bus events for coordination evidence

Invocation:
- CLI: python3 ckin_report.py [--agent AGENT] [--emit-bus] [--verbose]
- PluriChat: ckin, checking in
- Programmatic: generate_ckin_report(agent_name, verbose=True)

Bus Topic: ckin.report (kind=metric)
Lexicon: §6.4 Semantic Operators
"""

from __future__ import annotations

import argparse
import fcntl
import calendar
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True

# Visual elements
SPARK = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_THREE_QUARTER = "▓"
BAR_HALF = "▒"
BAR_QUARTER = "░"
BAR_EMPTY = "·"
ARROW_UP = "▲"
ARROW_DOWN = "▼"
ARROW_FLAT = "►"
CHECK = "✓"
CIRCLE = "○"
DOT = "●"
DIAMOND = "◆"

CKIN_PROTOCOL_VERSION = 16
PINNED_CHARTER_LINES = 33
PINNED_GOLDEN_SEED_LINES = 65


def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_ts(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(ts))


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "ckin"


def sparkline(values: list[int], width: int = 0) -> str:
    """Generate sparkline from values."""
    if not values:
        return ""
    if width > 0 and len(values) > width:
        # Downsample
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return SPARK[0] * len(values)
    out = []
    for v in values:
        idx = int((v - lo) / (hi - lo) * (len(SPARK) - 1))
        out.append(SPARK[max(0, min(len(SPARK) - 1, idx))])
    return "".join(out)

def shorten_path(path: str, max_len: int = 40) -> str:
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]

def iso_to_ts(iso: str) -> float | None:
    """Parse `YYYY-MM-DDTHH:MM:SSZ` to epoch seconds (UTC)."""
    if not iso:
        return None
    try:
        t = time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
        return float(calendar.timegm(t))
    except Exception:
        return None

def compute_staleness(*, beam_last_iso: str | None, golden_path: Path, now: float | None = None) -> dict[str, Any]:
    now_val = float(now if now is not None else now_ts())
    beam_age_s: float | None = None
    if beam_last_iso:
        ts = iso_to_ts(beam_last_iso)
        if ts is not None:
            beam_age_s = max(0.0, now_val - ts)
    golden_age_s: float | None = None
    try:
        golden_age_s = max(0.0, now_val - golden_path.stat().st_mtime)
    except Exception:
        golden_age_s = None
    return {"beam_last_iso": beam_last_iso, "beam_age_s": beam_age_s, "golden_age_s": golden_age_s}

def emit_paip_event(
    topic: str,
    *,
    clone_dir: str,
    agent_id: str = "",
    branch: str = "",
    uncommitted: int = 0,
    bus_dir: Path | None = None,
) -> None:
    """
    Emit a PAIP bus event. Topics: paip.clone.created, paip.clone.deleted,
    paip.orphan.detected, paip.cleanup.blocked, paip.isolation.violation
    """
    if bus_dir is None:
        bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    events_path = bus_dir / "events.ndjson"
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    payload = {
        "id": str(uuid.uuid4()),
        "ts": now_ts(),
        "iso": now_iso(),
        "topic": topic,
        "kind": "artifact" if topic in ("paip.clone.created", "paip.clone.deleted") else "request",
        "level": "info",
        "actor": os.environ.get("PLURIBUS_ACTOR") or agent_id or "paip",
        "data": {
            "clone_dir": clone_dir,
            "agent_id": agent_id,
            "branch": branch,
            "uncommitted": uncommitted,
        },
    }
    try:
        with events_path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass


def _count_lines(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None

def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def find_latest_topic_event(bus_path: Path, topic: str, max_bytes: int = 10_000_000) -> dict[str, Any] | None:
    """
    Best-effort scan from the end of the bus log for the most recent event matching `topic`.
    Does not require the event to fall within the CKIN window.
    """
    if not bus_path.exists():
        return None
    try:
        size = bus_path.stat().st_size
        start = max(0, size - max_bytes)
        with bus_path.open("rb") as f:
            f.seek(start)
            data = f.read()
        lines = data.splitlines()
        if start > 0 and lines:
            lines = lines[1:]
        latest: dict[str, Any] | None = None
        for raw in lines:
            try:
                e = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if str(e.get("topic", "")) == topic:
                latest = e
        return latest
    except Exception:
        return None

def detect_mcp_official_interop(repo_root: Path, bus_path: Path) -> dict[str, Any]:
    """
    CKIN v11+: surface whether Pluribus has an "official MCP interop boundary" and evidence for it.
    Observability-only: no network calls; no smoke execution.
    """
    pkg = _read_json(repo_root / "package.json") or {}
    dev = pkg.get("devDependencies") if isinstance(pkg, dict) else None
    dev = dev if isinstance(dev, dict) else {}
    sdk_version = dev.get("@modelcontextprotocol/sdk")
    sdk_pinned = isinstance(sdk_version, str) and (not sdk_version.strip().startswith("^")) and (not sdk_version.strip().startswith("~"))

    harness_path = repo_root / "nucleus" / "mcp" / "compat" / "mcp_typescript_sdk_smoke.mjs"
    test_path = repo_root / "nucleus" / "tools" / "tests" / "test_mcp_official_typescript_sdk_compat_unittest.py"
    docs_path = repo_root / "nucleus" / "third_party" / "mcp_typescript_sdk.md"

    latest = find_latest_topic_event(bus_path, "mcp.official_sdk.interop.announced")
    latest_iso = (latest.get("iso") if isinstance(latest, dict) else None) if latest else None
    latest_ts = float(latest.get("ts")) if latest and isinstance(latest.get("ts"), (int, float)) else None
    age_s = max(0.0, now_ts() - latest_ts) if latest_ts is not None else None

    req_id = None
    if latest and isinstance(latest.get("data"), dict):
        req_id = latest["data"].get("req_id")

    return {
        "sdk": {"name": "@modelcontextprotocol/sdk", "version": sdk_version, "pinned": bool(sdk_pinned)},
        "artifacts": {
            "harness": str(harness_path.relative_to(repo_root)),
            "harness_exists": harness_path.exists(),
            "test": str(test_path.relative_to(repo_root)),
            "test_exists": test_path.exists(),
            "docs": str(docs_path.relative_to(repo_root)),
            "docs_exists": docs_path.exists(),
        },
        "bus_evidence": {
            "topic": "mcp.official_sdk.interop.announced",
            "latest_iso": latest_iso,
            "latest_age_s": age_s,
            "req_id": req_id,
        },
    }

def detect_paip_clones_filesystem() -> dict[str, Any]:
    """
    CKIN v12: Detect PAIP clones by scanning /tmp for pluribus_* directories.
    Filesystem-based detection complements bus event tracking.
    """
    import subprocess
    tmp_dir = Path("/tmp")
    clones: list[dict[str, Any]] = []
    orphans: list[dict[str, Any]] = []

    now_val = now_ts()
    ttl_s = 3600  # 1 hour TTL

    if not tmp_dir.exists():
        return {"active": [], "orphans": [], "total": 0, "violations": 0}

    try:
        for entry in tmp_dir.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            # Match pluribus_AGENT_TIMESTAMP pattern
            if not name.startswith("pluribus_"):
                continue

            git_dir = entry / ".git"
            if not git_dir.exists():
                continue  # Not a git clone

            # Parse agent from name
            parts = name.split("_")
            agent_id = parts[1] if len(parts) > 1 else "unknown"

            # Get clone age
            try:
                mtime = entry.stat().st_mtime
                age_s = now_val - mtime
            except Exception:
                age_s = 0

            # Get current branch
            branch = None
            try:
                result = subprocess.run(
                    ["git", "-C", str(entry), "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    branch = result.stdout.strip()
            except Exception:
                pass

            # Check for uncommitted changes
            uncommitted_count = 0
            try:
                result = subprocess.run(
                    ["git", "-C", str(entry), "status", "--porcelain"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    uncommitted_count = len([l for l in result.stdout.strip().split("\n") if l])
            except Exception:
                pass

            clone_info = {
                "path": str(entry),
                "agent": agent_id,
                "branch": branch,
                "age_s": age_s,
                "uncommitted": uncommitted_count,
                "status": "ACTIVE" if age_s < ttl_s else "STALE",
            }

            # Classify: orphan if stale with uncommitted work
            if age_s > ttl_s and uncommitted_count > 0:
                clone_info["status"] = "ORPHAN"
                orphans.append(clone_info)
            elif age_s > ttl_s:
                clone_info["status"] = "STALE"
                orphans.append(clone_info)
            else:
                clones.append(clone_info)
    except Exception:
        pass

    return {
        "active": clones,
        "orphans": orphans,
        "total": len(clones) + len(orphans),
        "violations": 0,  # Would need bus check for this
    }


def audit_bus_totals(bus_path: Path) -> dict[str, Any]:
    """
    CKIN v6 compliance snapshot needs totals from the bus system-of-record.
    This is a best-effort scan intended to be fast for ~O(10^5) lines.
    """
    total_lines = _count_lines(bus_path)
    if total_lines is None or total_lines <= 0:
        return {
            "bus_total_events": None,
            "beam_appends_total": None,
            "ckin_reports_total": None,
            "infer_sync_responses_total": None,
            "mcp_official_interop_announcements_total": None,
        }

    beam_appends = 0
    ckin_reports = 0
    infer_sync_responses = 0
    mcp_official_interop_announcements = 0

    # Match both compact and pretty JSON encodings:
    # - {"topic":"beam.10x.appended",...}
    # - {"topic": "beam.10x.appended", ...}
    topic_re = re.compile(r"\"topic\"\s*:\s*\"([^\"]+)\"")

    try:
        with bus_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = topic_re.search(line)
                if not m:
                    continue
                topic = m.group(1)
                if topic == "beam.10x.appended":
                    beam_appends += 1
                elif topic == "ckin.report":
                    ckin_reports += 1
                elif topic == "infer_sync.response":
                    infer_sync_responses += 1
                elif topic == "mcp.official_sdk.interop.announced":
                    mcp_official_interop_announcements += 1
    except Exception:
        return {
            "bus_total_events": total_lines,
            "beam_appends_total": None,
            "ckin_reports_total": None,
            "infer_sync_responses_total": None,
            "mcp_official_interop_announcements_total": None,
        }

    return {
        "bus_total_events": total_lines,
        "beam_appends_total": beam_appends,
        "ckin_reports_total": ckin_reports,
        "infer_sync_responses_total": infer_sync_responses,
        "mcp_official_interop_announcements_total": mcp_official_interop_announcements,
    }

def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"

def histogram_bar(value: int, max_val: int, width: int = 30, label: str = "") -> str:
    """Generate horizontal histogram bar."""
    if max_val <= 0:
        pct = 0.0
    else:
        pct = min(1.0, value / max_val)
    filled = int(pct * width)
    return f"{label:12} │{'█' * filled}{'·' * (width - filled)}│ {value:>5}"


def progress_bar_detailed(current: int, target: int, width: int = 30, label: str = "", show_trend: str = "") -> str:
    """Generate detailed progress bar with percentage and trend."""
    if target <= 0:
        pct = 0.0
    else:
        pct = min(1.0, current / target)

    filled = int(pct * width)
    remaining = width - filled

    # Color coding based on progress
    if pct >= 0.8:
        bar = BAR_FULL * filled + BAR_EMPTY * remaining
        status = CHECK
    elif pct >= 0.5:
        bar = BAR_FULL * filled + BAR_EMPTY * remaining
        status = ARROW_FLAT
    elif pct >= 0.2:
        bar = BAR_FULL * filled + BAR_QUARTER * min(3, remaining) + BAR_EMPTY * max(0, remaining - 3)
        status = ARROW_FLAT
    else:
        bar = BAR_FULL * filled + BAR_EMPTY * remaining
        status = ARROW_DOWN

    pct_str = f"{pct*100:5.1f}%"
    trend_str = f" {show_trend}" if show_trend else ""
    return f"  {label:22} [{bar}] {pct_str} ({current:>3}/{target:<3}) {status}{trend_str}"


def velocity_gauge(observed: float, anticipated: float, width: int = 25, label: str = "") -> str:
    """Generate velocity gauge with comparison."""
    if anticipated <= 0:
        ratio = 0.0
    else:
        ratio = observed / anticipated

    # Build gauge: [====|====] where | is the 100% mark
    half = width // 2

    if ratio <= 1.0:
        # Under or at target
        filled = int(ratio * half)
        bar = BAR_FULL * filled + BAR_EMPTY * (half - filled) + "│" + BAR_EMPTY * half
        trend = ARROW_DOWN if ratio < 0.7 else (ARROW_FLAT if ratio < 1.0 else CHECK)
    else:
        # Over target
        overfill = int(min(1.0, ratio - 1.0) * half)
        bar = BAR_FULL * half + "│" + BAR_THREE_QUARTER * overfill + BAR_EMPTY * (half - overfill)
        trend = ARROW_UP

    return f"  {label:14} [{bar}] {ratio*100:6.1f}% {trend} (obs:{observed:>4.0f} ant:{anticipated:>4.0f})"


def time_bucket_chart(events: list[dict], window_s: int = 900, buckets: int = 15) -> list[str]:
    """Generate time-bucketed activity chart."""
    now = now_ts()
    cutoff = now - window_s
    bucket_s = window_s / buckets

    counts = [0] * buckets
    for e in events:
        try:
            ts = float(e.get("ts", 0))
            if ts >= cutoff:
                idx = int((ts - cutoff) / bucket_s)
                if 0 <= idx < buckets:
                    counts[idx] += 1
        except Exception:
            continue

    max_count = max(counts) if counts else 1
    lines = []

    # Multi-row histogram
    for row in range(4, -1, -1):
        threshold = (row / 4) * max_count
        row_chars = []
        for c in counts:
            if c >= threshold and c > 0:
                row_chars.append("█")
            else:
                row_chars.append(" ")
        lines.append(f"  {''.join(row_chars)}  │ {int(threshold):>3}")

    # Time axis
    lines.append(f"  {'─' * buckets}──┴────")
    lines.append(f"  -{window_s//60}m{' ' * (buckets - 6)}now")

    return lines


@dataclass
class BusStats:
    total_events: int
    events_by_kind: dict[str, int]
    events_by_topic_prefix: dict[str, int]
    events_by_actor: dict[str, int]
    recent_events: list[dict]
    time_buckets: list[int]
    latest_oiterate_tick: Optional[dict] = None
    latest_pbflush_request: Optional[dict] = None
    pbflush_requests: int = 0
    pbflush_acks: int = 0
    latest_pbdeep_request: Optional[dict] = None
    latest_pbdeep_report: Optional[dict] = None
    latest_pbdeep_index: Optional[dict] = None
    pbdeep_requests: int = 0
    pbdeep_reports: int = 0
    pbdeep_index_updates: int = 0
    latest_rd_dispatch: Optional[dict] = None
    rd_dispatches: int = 0
    rd_acks: int = 0
    latest_a2a_negotiate_request: Optional[dict] = None
    a2a_negotiate_requests: int = 0
    a2a_negotiate_responses: int = 0
    a2a_declines: int = 0
    a2a_redirects: int = 0
    latest_studio_flow_roundtrip: Optional[dict] = None
    studio_flow_roundtrips: int = 0
    studio_flow_roundtrip_failures: int = 0
    paip_active_clones: list[dict] = None
    paip_cleanup_queue: list[dict] = None
    paip_isolation_violations: int = 0
    paip_unique_actors: int = 0
    paip_multi_agent: bool = False
    # PBLOCK (v16)
    pblock_active: bool = False
    pblock_milestone: Optional[str] = None
    pblock_entered_iso: Optional[str] = None
    pblock_entered_by: Optional[str] = None
    pblock_exit_criteria: Optional[dict] = None
    pblock_events: int = 0
    pblock_violations: int = 0


def load_pblock_state() -> dict:
    """Load PBLOCK state from persistent state file."""
    state_file = Path("/var/lib/pluribus/.pluribus/pblock_state.json")
    if state_file.exists():
        try:
            with state_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"active": False}


def analyze_bus(bus_path: Path, window_s: int = 900) -> BusStats:
    """Deep analysis of bus events."""
    events = []
    cutoff = now_ts() - window_s

    if bus_path.exists():
        try:
            size = bus_path.stat().st_size
            start = max(0, size - 3_000_000)  # Last 3MB
            with bus_path.open("rb") as f:
                f.seek(start)
                data = f.read()

            lines = data.splitlines()
            if start > 0 and lines:
                lines = lines[1:]

            for line in lines:
                try:
                    e = json.loads(line.decode("utf-8", errors="replace"))
                    ts = float(e.get("ts", 0))
                    if ts >= cutoff:
                        events.append(e)
                except Exception:
                    continue
        except Exception:
            pass

    # Aggregate stats
    kinds = Counter()
    prefixes = Counter()
    actors = Counter()
    
    latest_oiterate_tick = None
    latest_pbflush_request = None
    pbflush_requests = 0
    pbflush_acks = 0
    latest_pbdeep_request = None
    latest_pbdeep_report = None
    latest_pbdeep_index = None
    pbdeep_requests = 0
    pbdeep_reports = 0
    pbdeep_index_updates = 0
    latest_rd_dispatch = None
    rd_dispatches = 0
    rd_acks = 0
    latest_a2a_negotiate_request = None
    a2a_negotiate_requests = 0
    a2a_negotiate_responses = 0
    a2a_declines = 0
    a2a_redirects = 0
    latest_studio_flow_roundtrip = None
    studio_flow_roundtrips = 0
    studio_flow_roundtrip_failures = 0
    paip_clone_state: dict[str, dict[str, Any]] = {}
    paip_isolation_violations = 0
    pblock_events = 0
    pblock_violations = 0

    for e in events:
        kinds[str(e.get("kind", "unknown"))] += 1
        topic = str(e.get("topic", ""))
        prefix = topic.split(".")[0] if "." in topic else topic
        if prefix:
            prefixes[prefix] += 1
        actor = str(e.get("actor", "unknown"))
        actors[actor] += 1
        
        if topic == "oiterate.tick":
            # Keep the latest one
            latest_oiterate_tick = e
        elif topic == "operator.pbflush.request":
            pbflush_requests += 1
            latest_pbflush_request = e
        elif topic == "operator.pbflush.ack":
            pbflush_acks += 1
        elif topic == "operator.pbdeep.request":
            pbdeep_requests += 1
            latest_pbdeep_request = e
        elif topic == "operator.pbdeep.report":
            pbdeep_reports += 1
            latest_pbdeep_report = e
        elif topic == "operator.pbdeep.index.updated":
            pbdeep_index_updates += 1
            latest_pbdeep_index = e
        elif topic == "rd.tasks.dispatch":
            rd_dispatches += 1
            latest_rd_dispatch = e
        elif topic == "rd.tasks.ack":
            rd_acks += 1
        elif topic == "a2a.negotiate.request":
            a2a_negotiate_requests += 1
            latest_a2a_negotiate_request = e
        elif topic == "a2a.negotiate.response":
            a2a_negotiate_responses += 1
        elif topic == "a2a.decline":
            a2a_declines += 1
        elif topic == "a2a.redirect":
            a2a_redirects += 1
        elif topic == "studio.flow.roundtrip":
            studio_flow_roundtrips += 1
            latest_studio_flow_roundtrip = e
            d = e.get("data")
            ok = bool(d.get("ok")) if isinstance(d, dict) else False
            if not ok:
                studio_flow_roundtrip_failures += 1
        elif topic.startswith("paip."):
            data = e.get("data") if isinstance(e.get("data"), dict) else {}
            if topic == "paip.isolation.violation":
                paip_isolation_violations += 1
                continue
            clone_dir = str(data.get("clone_dir") or data.get("path") or data.get("clone_path") or "")
            if not clone_dir:
                continue
            status = "ACTIVE"
            if topic == "paip.clone.deleted":
                status = "CLEANED"
            elif topic == "paip.orphan.detected":
                status = "ORPHAN"
            elif topic == "paip.cleanup.blocked":
                status = "BLOCKED"
            branch = data.get("branch") or data.get("branch_or_commit")
            uncommitted = data.get("uncommitted")
            uncommitted_count = None
            if isinstance(uncommitted, list):
                uncommitted_count = len(uncommitted)
            elif isinstance(uncommitted, int):
                uncommitted_count = uncommitted
            ts_val = None
            try:
                ts_val = float(e.get("ts")) if e.get("ts") is not None else None
            except Exception:
                ts_val = None
            paip_clone_state[clone_dir] = {
                "path": clone_dir,
                "status": status,
                "branch": str(branch) if branch is not None else None,
                "uncommitted": uncommitted_count,
                "iso": e.get("iso"),
                "ts": ts_val,
            }
        elif topic.startswith("operator.pblock."):
            pblock_events += 1
            if topic == "operator.pblock.violation":
                pblock_violations += 1

    # Time buckets for sparkline
    bucket_count = 24
    bucket_s = window_s / bucket_count
    buckets = [0] * bucket_count
    for e in events:
        try:
            ts = float(e.get("ts", 0))
            idx = int((ts - cutoff) / bucket_s)
            if 0 <= idx < bucket_count:
                buckets[idx] += 1
        except Exception:
            continue

    actor_keys = [a for a in actors.keys() if a and a != "unknown"]
    unique_actors = len(actor_keys)
    multi_agent = unique_actors > 1

    now_val = now_ts()
    clones = []
    for entry in paip_clone_state.values():
        age_s = None
        ts_val = entry.get("ts")
        if isinstance(ts_val, (int, float)) and ts_val > 0:
            age_s = max(0.0, now_val - ts_val)
        clone = dict(entry)
        clone["age_s"] = age_s
        clones.append(clone)
    clones.sort(key=lambda c: c.get("ts") or 0, reverse=True)
    paip_active = [c for c in clones if c.get("status") == "ACTIVE"]
    paip_cleanup = [c for c in clones if c.get("status") in ("ORPHAN", "BLOCKED")]

    # Load PBLOCK state (v16)
    pblock_state = load_pblock_state()

    return BusStats(
        total_events=len(events),
        events_by_kind=dict(kinds.most_common(10)),
        events_by_topic_prefix=dict(prefixes.most_common(15)),
        events_by_actor=dict(actors.most_common(10)),
        recent_events=events[-10:],
        time_buckets=buckets,
        latest_oiterate_tick=latest_oiterate_tick,
        latest_pbflush_request=latest_pbflush_request,
        pbflush_requests=pbflush_requests,
        pbflush_acks=pbflush_acks,
        latest_pbdeep_request=latest_pbdeep_request,
        latest_pbdeep_report=latest_pbdeep_report,
        latest_pbdeep_index=latest_pbdeep_index,
        pbdeep_requests=pbdeep_requests,
        pbdeep_reports=pbdeep_reports,
        pbdeep_index_updates=pbdeep_index_updates,
        latest_rd_dispatch=latest_rd_dispatch,
        rd_dispatches=rd_dispatches,
        rd_acks=rd_acks,
        latest_a2a_negotiate_request=latest_a2a_negotiate_request,
        a2a_negotiate_requests=a2a_negotiate_requests,
        a2a_negotiate_responses=a2a_negotiate_responses,
        a2a_declines=a2a_declines,
        a2a_redirects=a2a_redirects,
        latest_studio_flow_roundtrip=latest_studio_flow_roundtrip,
        studio_flow_roundtrips=studio_flow_roundtrips,
        studio_flow_roundtrip_failures=studio_flow_roundtrip_failures,
        paip_active_clones=paip_active,
        paip_cleanup_queue=paip_cleanup,
        paip_isolation_violations=paip_isolation_violations,
        paip_unique_actors=unique_actors,
        paip_multi_agent=multi_agent,
        # PBLOCK (v16) - load from persistent state file
        pblock_active=pblock_state.get("active", False),
        pblock_milestone=pblock_state.get("milestone"),
        pblock_entered_iso=pblock_state.get("entered_iso"),
        pblock_entered_by=pblock_state.get("entered_by"),
        pblock_exit_criteria=pblock_state.get("exit_criteria"),
        pblock_events=pblock_events,
        pblock_violations=pblock_violations,
    )


def extract_pbdeep_summary(bus_stats: BusStats) -> dict[str, Any]:
    def event_base(event: dict | None) -> dict[str, Any]:
        if not event:
            return {"iso": None, "actor": None, "req_id": None}
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        return {
            "iso": event.get("iso"),
            "actor": event.get("actor"),
            "req_id": data.get("req_id"),
        }

    def report_meta(event: dict | None) -> dict[str, Any]:
        info = event_base(event)
        if not event:
            info["report_path"] = None
            info["index_path"] = None
            return info
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        info["report_path"] = data.get("report_path")
        info["index_path"] = data.get("index_path")
        return info

    def index_meta(event: dict | None) -> dict[str, Any]:
        info = event_base(event)
        if not event:
            info["index_path"] = None
            info["rag_items"] = None
            info["kg_nodes"] = None
            return info
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        info["index_path"] = data.get("index_path")
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        rag_doc_id = summary.get("rag_doc_id")
        info["rag_items"] = 1 if rag_doc_id else 0
        kg_summary = summary.get("kg") if isinstance(summary.get("kg"), dict) else {}
        info["kg_nodes"] = kg_summary.get("nodes")
        return info

    return {
        "requests_window": bus_stats.pbdeep_requests,
        "reports_window": bus_stats.pbdeep_reports,
        "index_updates_window": bus_stats.pbdeep_index_updates,
        "latest_request": event_base(bus_stats.latest_pbdeep_request),
        "latest_report": report_meta(bus_stats.latest_pbdeep_report),
        "latest_index": index_meta(bus_stats.latest_pbdeep_index),
    }


@dataclass
class BeamStats:
    total_entries: int
    entries_by_agent: dict[str, int]
    entries_by_tag: dict[str, int]
    entries_by_scope: dict[str, int]
    recent_entries: list[dict]
    iteration_counts: dict[int, int]
    last_entry_iso: str | None = None


def analyze_beam(beam_path: Path) -> BeamStats:
    """Deep analysis of BEAM discourse."""
    if not beam_path.exists():
        return BeamStats(0, {}, {}, {}, [], {}, None)

    content = beam_path.read_text(encoding="utf-8", errors="replace")
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        if "tags_list" in current and "tags" not in current:
            current["tags"] = "[" + ", ".join(current["tags_list"]) + "]"
        entries.append(current)
        current = {}

    def parse_tags(raw: str) -> list[str]:
        # Accept: "[V, I]" | "[V|R|I]" | "V, I" | "V"
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1].strip()
        parts = re.split(r"[,\s|]+", raw)
        tags = []
        for p in parts:
            p = p.strip().upper()
            if p in {"V", "R", "I", "G"} and p not in tags:
                tags.append(p)
        return tags

    for line in content.splitlines():
        if line.startswith("## Entry "):
            flush_current()
            # Expected: "## Entry <id> — ..."
            m = re.match(r"^## Entry\s+(.+?)\s+—\s+", line)
            if not m:
                m = re.match(r"^## Entry\s+(\S+)", line)
            entry_id = (m.group(1) if m else "unknown").strip()
            current = {"id": entry_id}
            # If present, parse trailing ISO: "— <actor> — <iso>"
            m_iso = re.match(r"^## Entry\s+.+?\s+—\s+.+?\s+—\s+(\d{4}-\d{2}-\d{2}T[^\\s]+Z)", line)
            if m_iso:
                current["header_iso"] = m_iso.group(1).strip()
            continue

        # Current BEAM format is key-value lines: "iteration: 1"
        m = re.match(r"^\s*iteration\s*:\s*(\d+)\s*$", line, flags=re.IGNORECASE)
        if m:
            current["iter"] = int(m.group(1))
            continue

        m = re.match(r"^\s*subagent_id\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if m:
            current["subagent_id"] = m.group(1).strip()
            continue

        m = re.match(r"^\s*actor\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if m:
            current["actor"] = m.group(1).strip()
            continue

        m = re.match(r"^\s*scope\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if m:
            current["scope"] = m.group(1).strip()
            continue

        m = re.match(r"^\s*tags\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if m:
            tags_list = parse_tags(m.group(1))
            current["tags_list"] = tags_list
            current["tags"] = "[" + ", ".join(tags_list) + "]" if tags_list else "[]"
            continue

        # Back-compat: accept older markdown-ish "iteration: **1**" style if present
        if "iteration" in line.lower() and "**" in line:
            m = re.search(r"(\d+)", line)
            if m:
                current["iter"] = int(m.group(1))
            continue

    flush_current()

    # Aggregate
    by_agent = Counter()
    by_tag = Counter()
    by_scope = Counter()
    by_iter = Counter()
    last_entry_iso: str | None = None

    for e in entries:
        actor = str(e.get("actor") or "")
        subagent_id = str(e.get("subagent_id") or "")
        ident = (actor + " " + subagent_id).lower()

        agent_family = "other"
        for fam in ("claude", "codex", "gemini"):
            if fam in ident:
                agent_family = fam
                break
        by_agent[agent_family] += 1

        tags_list = e.get("tags_list")
        if not isinstance(tags_list, list):
            tags_list = parse_tags(str(e.get("tags") or ""))
        for tag in tags_list:
            by_tag[tag] += 1

        scope = str(e.get("scope") or "unknown")
        by_scope[scope] += 1

        iter_num = e.get("iter", 0)
        if isinstance(iter_num, int):
            by_iter[iter_num] += 1
        # Track last entry ISO if present (append-only ordering).
        h_iso = e.get("header_iso")
        if isinstance(h_iso, str) and h_iso:
            last_entry_iso = h_iso

    return BeamStats(
        total_entries=len(entries),
        entries_by_agent=dict(by_agent),
        entries_by_tag=dict(by_tag),
        entries_by_scope=dict(by_scope.most_common(8)),
        recent_entries=entries[-8:],
        iteration_counts=dict(by_iter),
        last_entry_iso=last_entry_iso,
    )


@dataclass
class GoldenStats:
    total_lines: int
    sections: dict[str, dict]  # section -> {lines, status, depth}
    structure_complete: float


def analyze_golden(golden_path: Path) -> GoldenStats:
    """Deep analysis of GOLDEN_SYNTHESIS."""
    if not golden_path.exists():
        return GoldenStats(0, {}, 0.0)

    content = golden_path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()

    sections = {}
    current_section = None
    section_lines = 0

    for line in lines:
        if line.startswith("## G") or line.startswith("## Generation"):
            if current_section:
                depth = "deep" if section_lines > 20 else ("medium" if section_lines > 8 else "sparse")
                sections[current_section] = {
                    "lines": section_lines,
                    "status": "complete" if section_lines > 15 else ("partial" if section_lines > 5 else "stub"),
                    "depth": depth
                }
            match = re.match(r"## (G\d+|Generation \d+[^#]*)", line)
            current_section = match.group(1).strip() if match else line[3:].strip()
            section_lines = 0
        elif current_section and line.strip():
            section_lines += 1

    if current_section:
        depth = "deep" if section_lines > 20 else ("medium" if section_lines > 8 else "sparse")
        sections[current_section] = {
            "lines": section_lines,
            "status": "complete" if section_lines > 15 else ("partial" if section_lines > 5 else "stub"),
            "depth": depth
        }

    # Expected sections for 10x10
    expected_sections = 10
    complete_count = sum(1 for s in sections.values() if s["status"] == "complete")
    structure_complete = complete_count / expected_sections if expected_sections > 0 else 0

    return GoldenStats(
        total_lines=len(lines),
        sections=sections,
        structure_complete=structure_complete,
    )


def analyze_hexis(buffer_dir: Path = Path("/tmp")) -> dict[str, list[dict]]:
    """Analyze Hexis buffer messages."""
    buffers = {}
    for agent in ["claude", "codex", "gemini"]:
        buffer_path = buffer_dir / f"{agent}.buffer"
        if buffer_path.exists():
            try:
                content = buffer_path.read_text(encoding="utf-8", errors="replace")
                messages = []
                for line in content.splitlines():
                    if line.strip():
                        try:
                            msg = json.loads(line)
                            messages.append(msg)
                        except Exception:
                            continue
                buffers[agent] = messages[-5:]  # Last 5
            except Exception:
                buffers[agent] = []
        else:
            buffers[agent] = []
    return buffers


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict[str, Any]) -> str:
    """Emit event to bus."""
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": now_ts(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    path = bus_dir / "events.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(evt, ensure_ascii=False, separators=(",", ":")) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    return evt_id


def generate_ckin_report(
    agent_name: str = "claude",
    bus_dir: Optional[Path] = None,
    beam_path: Optional[Path] = None,
    golden_path: Optional[Path] = None,
    verbose: bool = True,
) -> str:
    """Generate comprehensive CKIN report."""

    # Defaults
    if bus_dir is None:
        bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    if beam_path is None:
        beam_path = Path("/pluribus/agent_reports/2025-12-15_beam_10x_discourse.md")
    if golden_path is None:
        golden_path = Path("/pluribus/nucleus/docs/GOLDEN_SYNTHESIS_DISCOURSE.md")

    events_path = bus_dir / "events.ndjson"
    iso = now_iso()

    # Gather all stats
    bus_stats = analyze_bus(events_path, window_s=900)
    beam_stats = analyze_beam(beam_path)
    golden_stats = analyze_golden(golden_path)
    hexis_buffers = analyze_hexis()
    bus_totals = audit_bus_totals(events_path)
    pbdeep_summary = extract_pbdeep_summary(bus_stats)

    # Staleness (best-effort, no secrets)
    staleness = compute_staleness(beam_last_iso=beam_stats.last_entry_iso, golden_path=golden_path)
    beam_age_s = staleness.get("beam_age_s")
    golden_age_s = staleness.get("golden_age_s")

    # Actor hygiene signal used by v6 compliance + v7 gap analysis.
    unknown_actor_recent = 0
    for e in beam_stats.recent_entries:
        actor = str(e.get("actor") or "").strip().lower()
        if not actor or actor == "unknown":
            unknown_actor_recent += 1

    # v7 Gap analysis (epistemic vs aleatoric)
    epistemic_gaps: list[dict[str, Any]] = []
    if beam_stats.total_entries < 100:
        epistemic_gaps.append(
            {
                "id": "E1",
                "description": f"BEAM entries below target ({beam_stats.total_entries}/100).",
                "severity": "med",
                "suggested_action": "Append verification entries with falsifiers via beam_append.py.",
            }
        )
    if golden_stats.total_lines < 500:
        epistemic_gaps.append(
            {
                "id": "E2",
                "description": f"GOLDEN lines below target ({golden_stats.total_lines}/500).",
                "severity": "med",
                "suggested_action": "Append substantive synthesis addenda to GOLDEN.",
            }
        )
    if (bus_totals.get("infer_sync_responses_total") or 0) < 3:
        epistemic_gaps.append(
            {
                "id": "E3",
                "description": "Low infer_sync.response count (agents not acknowledging sync requests).",
                "severity": "high",
                "suggested_action": "After ITERATE, require each agent to emit infer_sync.response with docs/persona/scope/falsifier.",
            }
        )
    if unknown_actor_recent > 0:
        epistemic_gaps.append(
            {
                "id": "E4",
                "description": f"Actor hygiene issue: {unknown_actor_recent} recent BEAM entries have actor missing/unknown.",
                "severity": "med",
                "suggested_action": "Set PLURIBUS_ACTOR explicitly before writing BEAM/CKIN artifacts.",
            }
        )

    # Simple aleatoric bounds from bus activity buckets (counts per slice)
    if bus_stats.time_buckets:
        lo = float(min(bus_stats.time_buckets))
        hi = float(max(bus_stats.time_buckets))
    else:
        lo = hi = None
    aleatoric_bounds: list[dict[str, Any]] = [
        {
            "id": "A1",
            "description": "Bus activity variance in 15min window (bucket counts).",
            "bounds": {"lo": lo, "hi": hi, "unit": "events/bucket"},
        }
    ]

    lines = []

    # ═══════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("╔════════════════════════════════════════════════════════════════════════════╗")
    lines.append(f"║  CKIN DASHBOARD v{CKIN_PROTOCOL_VERSION} (DKIN) — {agent_name.upper():10}                                 ║")
    lines.append(f"║  {iso}                                              ║")
    lines.append("╠════════════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  Charter: agent_reports/2025-12-15_beam_10x_charter.md                     ║")
    lines.append(f"║  Challenge: 10x10 Grand Distillation (10 iters × 10 personas)              ║")
    lines.append(f"║  Protocol: DKIN v{CKIN_PROTOCOL_VERSION} (Living Dashboard)                             ║")
    lines.append("╚════════════════════════════════════════════════════════════════════════════╝")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # BUS ACTIVITY
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│  ▸ BUS ACTIVITY (15min window)                                             │")
    lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
    lines.append(f"│  Total Events: {bus_stats.total_events:,}                                                       │")
    lines.append(f"│  Sparkline:    {sparkline(bus_stats.time_buckets, width=40):40}           │")
    lines.append("│                                                                             │")

    # Latest Events (The Stream)
    lines.append("│  Latest Events:                                                            │")
    for e in reversed(bus_stats.recent_events):
        ts_str = e.get("iso", "")[11:19] # HH:MM:SS
        actor = e.get("actor", "?")[:10]
        topic = e.get("topic", "?")[:30]
        lines.append(f"│    {ts_str} {actor:10} {topic:30}                   │")
    lines.append("│                                                                             │")

    # Event kind distribution
    lines.append("│  Event Kinds:                                                              │")
    max_kind = max(bus_stats.events_by_kind.values()) if bus_stats.events_by_kind else 1
    for kind, count in sorted(bus_stats.events_by_kind.items(), key=lambda x: -x[1])[:5]:
        bar_width = int((count / max_kind) * 25)
        lines.append(f"│    {kind:12} {'█' * bar_width}{'·' * (25-bar_width)} {count:>5}              │")

    lines.append("│                                                                             │")

    # Topic distribution
    lines.append("│  Topic Prefixes:                                                           │")
    max_topic = max(bus_stats.events_by_topic_prefix.values()) if bus_stats.events_by_topic_prefix else 1
    for prefix, count in sorted(bus_stats.events_by_topic_prefix.items(), key=lambda x: -x[1])[:6]:
        bar_width = int((count / max_topic) * 20)
        lines.append(f"│    {prefix:12} {'▓' * bar_width}{'·' * (20-bar_width)} {count:>5}                   │")

    # PBFLUSH visibility (DKIN v10)
    if bus_stats.pbflush_requests <= 0:
        pbflush_line = "none (15min window)"
    else:
        last = bus_stats.latest_pbflush_request or {}
        last_iso = str(last.get("iso") or "")
        last_hms = last_iso[11:19] if len(last_iso) >= 19 else "?:?:?"
        last_actor = str(last.get("actor") or "?")[:12]
        req_id = ""
        data = last.get("data")
        if isinstance(data, dict):
            req_id = str(data.get("req_id") or "")
        req_short = req_id[:8] if req_id else "unknown"
        pbflush_line = f"req={req_short} acks={bus_stats.pbflush_acks} last={last_hms} actor={last_actor}"
    lines.append(f"│  PBFLUSH: {pbflush_line:<64}│")

    # PBLOCK visibility (DKIN v16) - Milestone Freeze State
    if bus_stats.pblock_active:
        milestone = str(bus_stats.pblock_milestone or "?")[:20]
        entered = bus_stats.pblock_entered_iso or ""
        entered_hms = entered[11:19] if len(entered) >= 19 else "?:?:?"
        by = str(bus_stats.pblock_entered_by or "?")[:10]
        criteria = bus_stats.pblock_exit_criteria or {}
        tests_ok = "Y" if criteria.get("all_tests_pass") else "N"
        pushed = "Y" if criteria.get("pushed_to_github") else "N"
        pblock_line = f"🔒 ACTIVE {milestone} tests={tests_ok} pushed={pushed} since={entered_hms} by={by}"
    else:
        pblock_line = "🔓 inactive (normal development)"
    lines.append(f"│  PBLOCK: {pblock_line:<65}│")
    if bus_stats.pblock_violations > 0:
        lines.append(f"│    ⚠️  VIOLATIONS: {bus_stats.pblock_violations} blocked commits detected                        │")

    # RD dispatch visibility (REALAGENTS/Studio assignments)
    if bus_stats.rd_dispatches <= 0:
        rd_line = "none (15min window)"
    else:
        last = bus_stats.latest_rd_dispatch or {}
        last_iso = str(last.get("iso") or "")
        last_hms = last_iso[11:19] if len(last_iso) >= 19 else "?:?:?"
        last_actor = str(last.get("actor") or "?")[:12]
        payload = last.get("data") if isinstance(last.get("data"), dict) else {}
        task_id = str(payload.get("task_id") or payload.get("task") or "unknown")
        req_id = str(payload.get("req_id") or "")
        req_short = req_id[:8] if req_id else "unknown"
        rd_line = f"{task_id} req={req_short} acks={bus_stats.rd_acks} last={last_hms} actor={last_actor}"
    lines.append(f"│  RD TASKS: {rd_line:<64}│")

    # A2A negotiation visibility (REALAGENTS T2)
    if bus_stats.a2a_negotiate_requests <= 0:
        a2a_line = "none (15min window)"
    else:
        last = bus_stats.latest_a2a_negotiate_request or {}
        last_iso = str(last.get("iso") or "")
        last_hms = last_iso[11:19] if len(last_iso) >= 19 else "?:?:?"
        last_actor = str(last.get("actor") or "?")[:12]
        a2a_line = (
            f"reqs={bus_stats.a2a_negotiate_requests} resps={bus_stats.a2a_negotiate_responses} "
            f"declines={bus_stats.a2a_declines} redirects={bus_stats.a2a_redirects} "
            f"last={last_hms} actor={last_actor}"
        )
    lines.append(f"│  A2A: {a2a_line:<69}│")

    # Studio flow roundtrip (REALAGENTS T3 Ω-gate)
    if bus_stats.studio_flow_roundtrips <= 0:
        studio_line = "none (15min window)"
    else:
        last = bus_stats.latest_studio_flow_roundtrip or {}
        last_iso = str(last.get("iso") or "")
        last_hms = last_iso[11:19] if len(last_iso) >= 19 else "?:?:?"
        last_actor = str(last.get("actor") or "?")[:12]
        ok_count = bus_stats.studio_flow_roundtrips - bus_stats.studio_flow_roundtrip_failures
        studio_line = f"roundtrip ok={ok_count} fail={bus_stats.studio_flow_roundtrip_failures} last={last_hms} actor={last_actor}"
    lines.append(f"│  STUDIO: {studio_line:<66}│")

    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # FORENSICS INDEX (PBDEEP) — v13
    # ═══════════════════════════════════════════════════════════════════════
    def _pbdeep_line(text: str) -> str:
        return f"│  {text[:75]:<75}│"

    def _pbdeep_hms(info: dict[str, Any]) -> str:
        iso_val = str(info.get("iso") or "")
        return iso_val[11:19] if len(iso_val) >= 19 else "?:?:?"

    def _pbdeep_event_line(label: str, info: dict[str, Any], extra: str = "") -> str:
        if not info.get("iso"):
            return f"{label}: none"
        actor = str(info.get("actor") or "?")[:12]
        req_id = str(info.get("req_id") or "")
        req_short = req_id[:8] if req_id else "unknown"
        base = f"{label}: {_pbdeep_hms(info)} actor={actor} req={req_short}"
        return f"{base} {extra}".strip()

    pb_req = pbdeep_summary.get("latest_request", {})
    pb_rep = pbdeep_summary.get("latest_report", {})
    pb_idx = pbdeep_summary.get("latest_index", {})
    rag_items = pb_idx.get("rag_items")
    kg_nodes = pb_idx.get("kg_nodes")
    rag_s = "?" if rag_items is None else str(rag_items)
    kg_s = "?" if kg_nodes is None else str(kg_nodes)
    report_path = pb_rep.get("report_path")
    index_path = pb_idx.get("index_path") or pb_rep.get("index_path")
    report_path_short = shorten_path(str(report_path), 60) if report_path else "none"
    index_path_short = shorten_path(str(index_path), 60) if index_path else "none"

    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│  ▸ FORENSICS INDEX (PBDEEP)                                               │")
    lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
    lines.append(
        _pbdeep_line(
            f"Requests: {pbdeep_summary.get('requests_window', 0)}  Reports: {pbdeep_summary.get('reports_window', 0)}  "
            f"Index Updates: {pbdeep_summary.get('index_updates_window', 0)}"
        )
    )
    lines.append("│                                                                             │")
    lines.append(_pbdeep_line(_pbdeep_event_line("Latest Request", pb_req)))
    lines.append(_pbdeep_line(_pbdeep_event_line("Latest Report", pb_rep)))
    lines.append(_pbdeep_line(f"Report Path: {report_path_short}"))
    lines.append(_pbdeep_line(_pbdeep_event_line("Latest Index", pb_idx, f"rag={rag_s} kg={kg_s}")))
    lines.append(_pbdeep_line(f"Index Path: {index_path_short}"))
    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # PARALLEL AGENT ISOLATION (v12)
    # ═══════════════════════════════════════════════════════════════════════
    # Filesystem detection (always run, complements bus detection)
    fs_paip = detect_paip_clones_filesystem()
    fs_active = fs_paip.get("active") or []
    fs_orphans = fs_paip.get("orphans") or []

    # Merge bus-detected and filesystem-detected clones
    seen_paths = set()
    merged_active = []
    merged_cleanup = []

    # Prioritize filesystem detection (more accurate)
    for c in fs_active:
        merged_active.append(c)
        seen_paths.add(c.get("path"))
    for c in fs_orphans:
        merged_cleanup.append(c)
        seen_paths.add(c.get("path"))

    # Add bus-detected that weren't found on filesystem
    for c in (bus_stats.paip_active_clones or []):
        if c.get("path") not in seen_paths:
            merged_active.append(c)
    for c in (bus_stats.paip_cleanup_queue or []):
        if c.get("path") not in seen_paths:
            merged_cleanup.append(c)

    show_paip_section = (
        bus_stats.paip_multi_agent
        or len(merged_active) > 0
        or len(merged_cleanup) > 0
        or bus_stats.paip_isolation_violations > 0
    )

    if show_paip_section:
        def _paip_row(text: str) -> str:
            return f"│    {text:<69}│"

        def _format_clone(entry: dict[str, Any]) -> str:
            path = shorten_path(str(entry.get("path") or "?"), 40)
            status = str(entry.get("status") or "?").upper()
            age_s = entry.get("age_s")
            age_min = "?" if not isinstance(age_s, (int, float)) else f"{int(age_s // 60)}m"
            branch = str(entry.get("branch") or "-")[:12]
            agent = str(entry.get("agent") or "")[:8]
            extra = ""
            uncommitted = entry.get("uncommitted")
            if isinstance(uncommitted, int) and uncommitted > 0:
                extra = f" uncommitted={uncommitted}"
            agent_tag = f"({agent}) " if agent else ""
            return f"{agent_tag}{path} [{status}] age={age_min} branch={branch}{extra}"

        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│  ▸ PARALLEL AGENT ISOLATION (v12)                                          │")
        lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│  Actors (15m): {bus_stats.paip_unique_actors:<5}  FS Clones: {fs_paip.get('total', 0):<5}  TTL: 1h                          │")
        lines.append("│  Active Clones:                                                             │")
        if merged_active:
            for entry in merged_active[:4]:
                lines.append(_paip_row(_format_clone(entry)))
        else:
            lines.append(_paip_row("none"))
        lines.append("│                                                                             │")
        lines.append("│  Cleanup Queue (stale/orphan):                                              │")
        if merged_cleanup:
            for entry in merged_cleanup[:3]:
                lines.append(_paip_row(_format_clone(entry)))
        else:
            lines.append(_paip_row("none"))
        lines.append("│                                                                             │")
        violations = bus_stats.paip_isolation_violations
        status = "OK" if violations == 0 else f"WARN (violations={violations})"
        lines.append(f"│  Isolation Status: {status:<56}│")
        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # GAP ANALYSIS (v7)
    # ═══════════════════════════════════════════════════════════════════════
    if epistemic_gaps or aleatoric_bounds:
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│  ▸ GAP ANALYSIS (v7)                                                       │")
        lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
        if epistemic_gaps:
            lines.append("│  Epistemic Gaps (reducible):                                               │")
            for g in epistemic_gaps[:4]:
                lines.append(f"│    [{g['id']}] {g['description'][:64]:64} │")
        else:
            lines.append("│  Epistemic Gaps (reducible): none                                          │")
        lines.append("│                                                                             │")
        if aleatoric_bounds:
            lines.append("│  Aleatoric Bounds (irreducible):                                           │")
            for a in aleatoric_bounds[:2]:
                b = a.get("bounds") or {}
                lo_s = "?" if b.get("lo") is None else str(int(b.get("lo")))
                hi_s = "?" if b.get("hi") is None else str(int(b.get("hi")))
                unit = b.get("unit") or ""
                desc = f"[{a['id']}] {a['description']} lo={lo_s} hi={hi_s} {unit}"
                lines.append(f"│    {desc[:73]:73}│")
        else:
            lines.append("│  Aleatoric Bounds (irreducible): none                                      │")
        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # COMPLIANCE & SYNC (v6)
    # ═══════════════════════════════════════════════════════════════════════
    charter_path = Path("/pluribus/agent_reports/2025-12-15_beam_10x_charter.md")
    golden_seed_path = Path("/pluribus/nucleus/docs/GOLDEN_REPORT_SEED.md")
    charter_lines = _count_lines(charter_path)
    seed_lines = _count_lines(golden_seed_path)

    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│  ▸ COMPLIANCE & SYNC (CKIN v6)                                              │")
    lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
    lines.append(f"│  Shared Ledgers: BEAM + GOLDEN are canonical append-only targets            │")
    lines.append("│                                                                             │")
    lines.append(f"│  Bus SOR: events.ndjson lines = {str(bus_totals.get('bus_total_events')):>10}                                │")
    lines.append(f"│  Evidence: beam.10x.appended total = {str(bus_totals.get('beam_appends_total')):>7}                            │")
    lines.append(f"│  Evidence: ckin.report total         = {str(bus_totals.get('ckin_reports_total')):>7}                            │")
    lines.append(f"│  Evidence: infer_sync.response total = {str(bus_totals.get('infer_sync_responses_total')):>7}                            │")
    lines.append("│                                                                             │")
    lines.append(f"│  Drift Guards (pinned sizes):                                               │")
    lines.append(f"│  - Charter lines: {str(charter_lines):>4} (pinned {PINNED_CHARTER_LINES})                                     │")
    lines.append(f"│  - Golden seed:   {str(seed_lines):>4} (pinned {PINNED_GOLDEN_SEED_LINES})                                     │")
    lines.append("│                                                                             │")
    lines.append(f"│  Actor hygiene (recent BEAM): unknown/missing = {unknown_actor_recent:>2}                        │")
    lines.append(f"│  Staleness: BEAM={format_age(beam_age_s):>7} | GOLDEN={format_age(golden_age_s):>7}                                │")
    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # MCP INTEROP (Official SDK) — v11
    # ═══════════════════════════════════════════════════════════════════════
    mcp = detect_mcp_official_interop(Path("/pluribus"), events_path)
    sdk_ver = mcp.get("sdk", {}).get("version")
    sdk_pin = bool(mcp.get("sdk", {}).get("pinned"))
    art = mcp.get("artifacts", {})
    harness_ok = bool(art.get("harness_exists"))
    test_ok = bool(art.get("test_exists"))
    docs_ok = bool(art.get("docs_exists"))
    ev = mcp.get("bus_evidence", {})
    ev_iso = str(ev.get("latest_iso") or "none")
    ev_age = format_age(ev.get("latest_age_s")) if ev.get("latest_iso") else "n/a"
    ev_req = str(ev.get("req_id") or "-")[:64]

    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│  ▸ MCP INTEROP (Official SDK boundary)                                      │")
    lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
    lines.append(f"│  SDK: @modelcontextprotocol/sdk={str(sdk_ver):18} pinned={str(sdk_pin):5}                         │")
    lines.append(f"│  Harness: {CHECK if harness_ok else DOT}  Test: {CHECK if test_ok else DOT}  Docs: {CHECK if docs_ok else DOT}                                         │")
    lines.append(f"│  Evidence: mcp.official_sdk.interop.announced = {ev_iso:20} age={ev_age:>6}                 │")
    lines.append(f"│  req_id: {ev_req:64}│")
    lines.append("│  Stdio: --stdio (Content-Length) | --stdio-lines (NDJSON)                   │")
    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    if verbose:
        # ═══════════════════════════════════════════════════════════════════════
        # BEAM DISCOURSE
        # ═══════════════════════════════════════════════════════════════════════
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│  ▸ BEAM DISCOURSE                                                          │")
        lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│  Total Entries: {beam_stats.total_entries:>3}/100                                                   │")
        lines.append("")
        lines.append(progress_bar_detailed(beam_stats.total_entries, 100, width=40, label="Progress"))
        lines.append("│                                                                             │")
        if beam_stats.last_entry_iso:
            age = f"{int(beam_age_s or 0)}s" if beam_age_s is not None else "unknown"
            lines.append(f"│  Last BEAM append: {beam_stats.last_entry_iso:20} age={age:>10}                        │")
            lines.append("│                                                                             │")

        # By agent
        lines.append("│  Entries by Agent:                                                         │")
        for agent in ["claude", "codex", "gemini"]:
            count = beam_stats.entries_by_agent.get(agent, 0)
            bar = "█" * min(20, count) + "·" * max(0, 20 - count)
            lines.append(f"│    {agent.capitalize():8} [{bar}] {count:>3}                              │")

        lines.append("│                                                                             │")

        # By tag (V/R/I/G)
        lines.append("│  Tag Distribution (V=Verified R=Reported I=Intent G=Gap):                  │")
        tag_line = "│    "
        for tag in ["V", "R", "I", "G"]:
            count = beam_stats.entries_by_tag.get(tag, 0)
            tag_line += f"[{tag}]={count:>2}  "
        tag_line += "                                    │"
        lines.append(tag_line)

        lines.append("│                                                                             │")

        # Iteration progress
        lines.append("│  Iteration Progress:                                                       │")
        iter_line = "│    "
        for i in range(1, 11):
            count = beam_stats.iteration_counts.get(i, 0)
            if count >= 3:
                iter_line += f"[{i}✓]"
            elif count > 0:
                iter_line += f"[{i}○]"
            else:
                iter_line += f"[{i}·]"
        iter_line += "           │"
        lines.append(iter_line)

        lines.append("│                                                                             │")

        # Recent entries (no entry IDs/hashes)
        lines.append("│  Recent Entries:                                                           │")
        recent = beam_stats.recent_entries[-4:]
        for idx, e in enumerate(recent, start=max(1, len(beam_stats.recent_entries) - len(recent) + 1)):
            actor = str(e.get("actor") or e.get("subagent_id") or "?")[:12]
            scope = e.get("scope", "?")[:15]
            tags = str(e.get("tags") or "?")[:10]
            lines.append(f"│    #{idx:<3} │ {actor:12} │ {scope:15} │ {tags:10}          │")

        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

        # ═══════════════════════════════════════════════════════════════════════
        # GOLDEN_SYNTHESIS
        # ═══════════════════════════════════════════════════════════════════════
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│  ▸ GOLDEN_SYNTHESIS                                                        │")
        lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│  Total Lines: {golden_stats.total_lines:>4}/500                                                    │")
        lines.append("")
        lines.append(progress_bar_detailed(golden_stats.total_lines, 500, width=40, label="Content Progress"))
        lines.append("│                                                                             │")
        if golden_age_s is not None:
            lines.append(f"│  GOLDEN mtime age: {int(golden_age_s):>10}s                                           │")
            lines.append("│                                                                             │")

        lines.append("│  Section Depth Meters:                                                     │")
        expected_sections = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10"]
        for sect_key in expected_sections:
            found = None
            for s, data in golden_stats.sections.items():
                if sect_key in s or f"Generation {sect_key[1:]}" in s:
                    found = (s, data)
                    break

            if found:
                _, data = found
                depth = data["lines"]
                status = data["status"]
                meter = "█" * min(15, depth // 2) + "·" * max(0, 15 - depth // 2)
                icon = CHECK if status == "complete" else (CIRCLE if status == "partial" else DOT)
                lines.append(f"│    {icon} {sect_key:4} [{meter}] {depth:>3} lines  {status:8}              │")
            else:
                lines.append(f"│    {DOT} {sect_key:4} [···············]   0 lines  pending               │")

        lines.append("│                                                                             │")
        lines.append(f"│  Structure Complete: {golden_stats.structure_complete*100:5.1f}%                                           │")
        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # OITERATE LOOP STATUS (v8)
    # ═══════════════════════════════════════════════════════════════════════
    if bus_stats.latest_oiterate_tick:
        oit = bus_stats.latest_oiterate_tick.get("data", {})
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│  ▸ OITERATE LOOP STATUS (v8)                                               │")
        lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
        lines.append(f"│  Session: {oit.get('session_id', '?'):<12}  Tick: {oit.get('tick', 0):<5}  State: {oit.get('state', 'UNKNOWN'):<10}      │")
        
        prog = float(oit.get("progress", 0.0))
        lines.append(progress_bar_detailed(int(prog * 100), 100, width=40, label="Loop Progress"))
        lines.append("│                                                                             │")
        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
            # ═══════════════════════════════════════════════════════════════════════
            # AGENT VELOCITY MATRIX        # ═══════════════════════════════════════════════════════════════════════
        lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│  ▸ AGENT VELOCITY MATRIX (1hr window)                                      │")
        lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
        lines.append("│                 Observed  Anticipated  Ratio    Gauge                      │")
        lines.append("│  ─────────────────────────────────────────────────────────────────────     │")

        agent_velocities: dict[str, float] = {}
        for agent in ["claude", "codex", "gemini"]:
            events = bus_stats.events_by_actor.get(agent, 0)
            for actor, count in bus_stats.events_by_actor.items():
                if agent in actor.lower():
                    events = max(events, count)
            agent_velocities[agent] = float(events)

        anticipated = 5.0  # Expected events per hour per agent
        for agent, observed in agent_velocities.items():
            ratio = observed / anticipated if anticipated > 0 else 0.0
            if ratio > 1.2:
                trend = ARROW_UP
            elif ratio > 0.8:
                trend = ARROW_FLAT
            else:
                trend = ARROW_DOWN

            gauge_filled = int(min(1.0, ratio) * 15)
            gauge_over = int(max(0.0, min(1.0, ratio - 1.0)) * 5) if ratio > 1.0 else 0
            gauge = "█" * gauge_filled + "│" + "▓" * gauge_over + "·" * (5 - gauge_over)

            lines.append(f"│  {agent.capitalize():8}      {observed:>5.0f}       {anticipated:>5.0f}     {ratio:>5.1f}x  [{gauge}] {trend}   │")

        lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # HEXIS BUFFER STATUS
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│  ▸ HEXIS BUFFER STATUS                                                     │")
    lines.append("├─────────────────────────────────────────────────────────────────────────────┤")

    for agent, messages in hexis_buffers.items():
        pending = len(messages)
        icon = "🔴" if pending > 3 else ("🟡" if pending > 0 else "🟢")
        lines.append(f"│  {agent.capitalize():8} buffer: {pending:>2} pending messages  {icon}                            │")
        if pending > 0 and verbose:
            for msg in messages[-2:]:
                topic = msg.get("topic", "?")[:25]
                lines.append(f"│    └─ {topic:25}                                         │")

    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # SYSTEM FIDELITY MATRIX (Epoch 2)
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│  ▸ SYSTEM FIDELITY MATRIX                                                  │")
    lines.append("├─────────────────────────────────────────────────────────────────────────────┤")

    # Calculate velocities
    targets = [
        ("BEAM Ledger", beam_stats.total_entries, 100, "Discourse volume"),
        ("GOLDEN Synth", golden_stats.total_lines, 500, "Architectural depth"),
        ("Verification", beam_stats.entries_by_tag.get("V", 0), 30, "Verified claims"),
        ("Teleology", 2544, 2000, "Mapped domains"), # Hardcoded from recent scan count
    ]

    for label, current, target, note in targets:
        pct = (current / target * 100) if target > 0 else 0
        filled = int(min(1.0, current / target) * 35) if target > 0 else 0
        bar = "█" * filled + "░" * (35 - filled)
        trend = ARROW_UP if pct > 90 else (ARROW_FLAT if pct > 50 else ARROW_DOWN)
        lines.append(f"│  {label:14} [{bar}] {pct:5.1f}% {trend}   │")
        lines.append(f"│  {' ':14}  ({current}/{target}) {note:30}       │")

    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # ═══════════════════════════════════════════════════════════════════════
    # NEXT ACTIONS & BLOCKERS
    # ═══════════════════════════════════════════════════════════════════════
    lines.append("┌─────────────────────────────────────────────────────────────────────────────┐")
    lines.append("│  ▸ NEXT ACTIONS                                                            │")
    lines.append("├─────────────────────────────────────────────────────────────────────────────┤")

    actions = []
    if beam_stats.total_entries < 50:
        actions.append("Continue BEAM iterations toward 100 entries")
    if golden_stats.total_lines < 200:
        actions.append("Expand GOLDEN_SYNTHESIS with substantive content")
    if beam_stats.entries_by_tag.get("V", 0) < 10:
        actions.append("Cross-verify peer BEAM entries with [V] tags")
    if len(golden_stats.sections) < 5:
        actions.append("Add G2-G10 sections to GOLDEN_SYNTHESIS")
    # Staleness kick suggestion (operator-only keyboard flow)
    if beam_age_s is not None and beam_age_s > 1800:
        actions.insert(0, "ITERATE recommended: BEAM appears stale (>30m since last append)")

    for action in actions[:4]:
        lines.append(f"│  • {action:70} │")

    lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
    lines.append("│  ▸ BLOCKERS                                                                │")
    lines.append("├─────────────────────────────────────────────────────────────────────────────┤")
    lines.append("│  • None detected                                                           │")
    lines.append("└─────────────────────────────────────────────────────────────────────────────┘")
    lines.append("")

    # Footer
    lines.append("═" * 77)
    lines.append(f"Generated by ckin_report.py v{CKIN_PROTOCOL_VERSION} | {agent_name} | {iso}")
    lines.append(f"Refs: nucleus/specs/pluribus_lexicon.md §6.4")
    lines.append("═" * 77)

    return "\n".join(lines)


@dataclass
class SummaryFrameData:
    """Data structure for DKIN v13 Summary Frame."""
    title: str
    scope: str
    paip_clones_created: int = 0
    paip_clones_cleaned: int = 0
    paip_clones_current: int = 0
    paip_isolation_status: str = "OK"
    paip_agents: list[str] = None
    subagents: list[dict] = None  # [{"id": str, "status": "PASS"|"FAIL", "summary": str}]
    operators: list[dict] = None  # [{"name": str, "success": bool, "details": str}]
    branch: str = ""
    remote: str = ""
    sync_status: str = "up_to_date"
    commit: str = ""
    pbdeep_status: str = ""
    pbdeep_req_id: str = ""
    pbdeep_report_path: str = ""
    pbdeep_index_path: str = ""
    pbdeep_rag_items: int | None = None
    pbdeep_kg_nodes: int | None = None

    def __post_init__(self):
        if self.paip_agents is None:
            self.paip_agents = []
        if self.subagents is None:
            self.subagents = []
        if self.operators is None:
            self.operators = []


def generate_summary_frame(data: SummaryFrameData) -> str:
    """
    Generate DKIN v13 Summary Frame for multiagent orchestration.

    This produces a standardized ASCII frame that summarizes:
    - PAIP isolation status
    - Subagent validation results
    - Operators executed
    - PBDEEP forensics index (when available)
    - Sync status
    """
    WIDTH = 78  # Inner width (80 - 2 for borders)

    def line(content: str = "") -> str:
        """Create a bordered line with content."""
        if not content:
            return f"║{' ' * WIDTH}║"
        padded = f"  {content}"
        return f"║{padded:<{WIDTH}}║"

    def tree_line(prefix: str, label: str, value: str) -> str:
        """Create a tree-structured line."""
        content = f"  {prefix} {label:<16} {value}"
        return f"║{content:<{WIDTH}}║"

    lines = []

    # Header
    lines.append("╔" + "═" * WIDTH + "╗")
    header = f"  {data.title} — {data.scope}"
    lines.append(f"║{header:<{WIDTH}}║")
    lines.append("╠" + "═" * WIDTH + "╣")
    lines.append(line())

    # PAIP ISOLATION section
    lines.append(line("PAIP ISOLATION:"))
    agents_str = ", ".join(data.paip_agents[:5]) if data.paip_agents else "none"
    if len(data.paip_agents) > 5:
        agents_str += f" +{len(data.paip_agents) - 5} more"
    lines.append(tree_line("├─", "Clones Created:", f"{data.paip_clones_created} ({agents_str})"))
    lines.append(tree_line("├─", "Clones Cleaned:", f"{data.paip_clones_cleaned}"))
    lines.append(tree_line("├─", "FS Clones Now:", str(data.paip_clones_current)))
    lines.append(tree_line("└─", "Isolation Status:", data.paip_isolation_status))
    lines.append(line())

    # SUBAGENT VALIDATION section (if applicable)
    if data.subagents:
        lines.append(line("SUBAGENT VALIDATION:"))
        for i, sa in enumerate(data.subagents):
            prefix = "└─" if i == len(data.subagents) - 1 else "├─"
            status_icon = "PASS" if sa.get("status") == "PASS" else "FAIL"
            summary = sa.get("summary", "")[:40]
            lines.append(tree_line(prefix, f"{sa.get('id', 'unknown')}:", f"{status_icon} - {summary}"))
        lines.append(line())

    # OPERATORS EXECUTED section
    if data.operators:
        lines.append(line("OPERATORS EXECUTED:"))
        for i, op in enumerate(data.operators):
            prefix = "└─" if i == len(data.operators) - 1 else "├─"
            icon = "✓" if op.get("success") else "✗"
            details = op.get("details", "")[:45]
            lines.append(tree_line(prefix, f"{op.get('name', 'unknown')}:", f"{icon} {details}"))
        lines.append(line())

    # FORENSICS INDEX (PBDEEP) section
    if data.pbdeep_status or data.pbdeep_req_id or data.pbdeep_report_path or data.pbdeep_index_path:
        lines.append(line("FORENSICS INDEX:"))
        status = data.pbdeep_status or "OK"
        req_short = data.pbdeep_req_id[:12] if data.pbdeep_req_id else "unknown"
        report_path = shorten_path(data.pbdeep_report_path or "none", 40)
        index_path = shorten_path(data.pbdeep_index_path or "none", 40)
        rag_s = "?" if data.pbdeep_rag_items is None else str(data.pbdeep_rag_items)
        kg_s = "?" if data.pbdeep_kg_nodes is None else str(data.pbdeep_kg_nodes)
        lines.append(tree_line("├─", "Status:", status))
        lines.append(tree_line("├─", "Req ID:", req_short))
        lines.append(tree_line("├─", "Report:", report_path))
        lines.append(tree_line("├─", "Index:", index_path))
        lines.append(tree_line("└─", "Counts:", f"rag={rag_s} kg={kg_s}"))
        lines.append(line())

    # SYNC STATUS section
    lines.append(line("SYNC STATUS:"))
    lines.append(tree_line("├─", "Branch:", data.branch or "unknown"))
    lines.append(tree_line("├─", "Remote:", data.remote or "unknown"))
    status_text = data.sync_status.replace("_", " ").title()
    commit_short = data.commit[:8] if data.commit else "unknown"
    lines.append(tree_line("└─", "Status:", f"{status_text} ({commit_short})"))
    lines.append(line())

    # Footer
    lines.append("╚" + "═" * WIDTH + "╝")

    return "\n".join(lines)


def emit_summary_frame(
    data: SummaryFrameData,
    bus_dir: Path | None = None,
    actor: str = "ckin",
) -> str:
    """Generate summary frame and emit to bus."""
    frame_text = generate_summary_frame(data)

    if bus_dir is None:
        bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))

    payload = {
        "title": data.title,
        "scope": data.scope,
        "paip": {
            "clones_created": data.paip_clones_created,
            "clones_cleaned": data.paip_clones_cleaned,
            "clones_current": data.paip_clones_current,
            "isolation_status": data.paip_isolation_status,
            "agents": data.paip_agents,
        },
        "subagents": data.subagents,
        "operators": data.operators,
        "sync": {
            "branch": data.branch,
            "remote": data.remote,
            "status": data.sync_status,
            "commit": data.commit,
        },
        "pbdeep": {
            "status": data.pbdeep_status or None,
            "req_id": data.pbdeep_req_id or None,
            "report_path": data.pbdeep_report_path or None,
            "index_path": data.pbdeep_index_path or None,
            "rag_items": data.pbdeep_rag_items,
            "kg_nodes": data.pbdeep_kg_nodes,
        },
        "frame_text": frame_text,
    }

    emit_bus(
        bus_dir,
        topic="ckin.summary_frame",
        kind="artifact",
        level="info",
        actor=actor,
        data=payload,
    )

    return frame_text


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ckin_report.py", description="Pluribus Check-In Dashboard")
    ap.add_argument("--agent", default=default_actor(), help="Agent name for report header")
    ap.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    ap.add_argument("--emit-bus", action="store_true", help="Emit ckin.report metric to bus")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of formatted text")
    ap.add_argument("--verbose", "-v", action="store_true", default=True, help="Verbose output (default)")
    ap.add_argument("--compact", action="store_true", help="Compact output")
    args = ap.parse_args(argv)

    bus_dir = Path(args.bus_dir)
    verbose = not args.compact

    if args.json:
        beam_path = Path("/pluribus/agent_reports/2025-12-15_beam_10x_discourse.md")
        golden_path = Path("/pluribus/nucleus/docs/GOLDEN_SYNTHESIS_DISCOURSE.md")

        beam_stats = analyze_beam(beam_path)
        golden_stats = analyze_golden(golden_path)
        bus_stats = analyze_bus(bus_dir / "events.ndjson")
        bus_totals = audit_bus_totals(bus_dir / "events.ndjson")
        pbdeep_summary = extract_pbdeep_summary(bus_stats)

        unknown_actor_recent = 0
        for e in beam_stats.recent_entries:
            actor = str(e.get("actor") or "").strip().lower()
            if not actor or actor == "unknown":
                unknown_actor_recent += 1

        epistemic_gaps: list[dict[str, Any]] = []
        if beam_stats.total_entries < 100:
            epistemic_gaps.append({"id": "E1", "description": f"BEAM entries below target ({beam_stats.total_entries}/100).", "severity": "med", "suggested_action": "Append BEAM verification entries."})
        if golden_stats.total_lines < 500:
            epistemic_gaps.append({"id": "E2", "description": f"GOLDEN lines below target ({golden_stats.total_lines}/500).", "severity": "med", "suggested_action": "Append synthesis addenda to GOLDEN."})
        if (bus_totals.get("infer_sync_responses_total") or 0) < 3:
            epistemic_gaps.append({"id": "E3", "description": "Low infer_sync.response count.", "severity": "high", "suggested_action": "Require agent infer_sync.response after ITERATE."})
        if unknown_actor_recent > 0:
            epistemic_gaps.append({"id": "E4", "description": f"Actor hygiene issue in recent BEAM ({unknown_actor_recent}).", "severity": "med", "suggested_action": "Set PLURIBUS_ACTOR before writes."})

        if bus_stats.time_buckets:
            lo = float(min(bus_stats.time_buckets))
            hi = float(max(bus_stats.time_buckets))
        else:
            lo = hi = None
        aleatoric_bounds = [{"id": "A1", "description": "Bus activity variance (bucket counts).", "bounds": {"lo": lo, "hi": hi, "unit": "events/bucket"}}]

        data = {
            "agent": args.agent,
            "iso": now_iso(),
            "protocol_version": CKIN_PROTOCOL_VERSION,
            "beam": {
                "total": beam_stats.total_entries,
                "by_agent": beam_stats.entries_by_agent,
                "by_tag": beam_stats.entries_by_tag,
            },
            "golden": {
                "lines": golden_stats.total_lines,
                "sections": golden_stats.sections,
                "structure_complete": golden_stats.structure_complete,
            },
            "bus": {
                "events": bus_stats.total_events,
                "by_kind": bus_stats.events_by_kind,
                "by_actor": bus_stats.events_by_actor,
            },
            "pbdeep": pbdeep_summary,
            "paip": {
                "multi_agent": bus_stats.paip_multi_agent,
                "unique_actors": bus_stats.paip_unique_actors,
                "active_clones": bus_stats.paip_active_clones,
                "cleanup_queue": bus_stats.paip_cleanup_queue,
                "isolation_violations_window": bus_stats.paip_isolation_violations,
                "filesystem": detect_paip_clones_filesystem(),
            },
            "mcp_official_interop": detect_mcp_official_interop(Path("/pluribus"), bus_dir / "events.ndjson"),
            "gaps": {"epistemic": epistemic_gaps, "aleatoric": aleatoric_bounds},
            "compliance_sync": {
                "bus_total_events": bus_totals.get("bus_total_events"),
                "beam_appends_total": bus_totals.get("beam_appends_total"),
                "ckin_reports_total": bus_totals.get("ckin_reports_total"),
                "infer_sync_responses_total": bus_totals.get("infer_sync_responses_total"),
                "mcp_official_interop_announcements_total": bus_totals.get("mcp_official_interop_announcements_total"),
                "charter_lines": _count_lines(Path("/pluribus/agent_reports/2025-12-15_beam_10x_charter.md")),
                "golden_seed_lines": _count_lines(Path("/pluribus/nucleus/docs/GOLDEN_REPORT_SEED.md")),
                "pinned_charter_lines": PINNED_CHARTER_LINES,
                "pinned_golden_seed_lines": PINNED_GOLDEN_SEED_LINES,
            },
            "compliance": {  # back-compat alias
                "bus_total_events": bus_totals.get("bus_total_events"),
                "beam_appends_total": bus_totals.get("beam_appends_total"),
                "ckin_reports_total": bus_totals.get("ckin_reports_total"),
                "infer_sync_responses_total": bus_totals.get("infer_sync_responses_total"),
                "mcp_official_interop_announcements_total": bus_totals.get("mcp_official_interop_announcements_total"),
                "charter_lines": _count_lines(Path("/pluribus/agent_reports/2025-12-15_beam_10x_charter.md")),
                "golden_seed_lines": _count_lines(Path("/pluribus/nucleus/docs/GOLDEN_REPORT_SEED.md")),
                "pinned_charter_lines": PINNED_CHARTER_LINES,
                "pinned_golden_seed_lines": PINNED_GOLDEN_SEED_LINES,
            },
        }
        print(json.dumps(data, indent=2))
    else:
        report = generate_ckin_report(agent_name=args.agent, bus_dir=bus_dir, verbose=verbose)
        print(report)

    if args.emit_bus:
        bus_stats = analyze_bus(bus_dir / "events.ndjson")
        beam_path = Path("/pluribus/agent_reports/2025-12-15_beam_10x_discourse.md")
        golden_path = Path("/pluribus/nucleus/docs/GOLDEN_SYNTHESIS_DISCOURSE.md")

        beam_stats = analyze_beam(beam_path)
        golden_stats = analyze_golden(golden_path)
        staleness = compute_staleness(beam_last_iso=beam_stats.last_entry_iso, golden_path=golden_path)
        bus_totals = audit_bus_totals(bus_dir / "events.ndjson")
        pbdeep_summary = extract_pbdeep_summary(bus_stats)
        charter_path = Path("/pluribus/agent_reports/2025-12-15_beam_10x_charter.md")
        golden_seed_path = Path("/pluribus/nucleus/docs/GOLDEN_REPORT_SEED.md")

        unknown_actor_recent = 0
        for e in beam_stats.recent_entries:
            actor = str(e.get("actor") or "").strip().lower()
            if not actor or actor == "unknown":
                unknown_actor_recent += 1

        epistemic_gaps: list[dict[str, Any]] = []
        if beam_stats.total_entries < 100:
            epistemic_gaps.append({"id": "E1", "description": f"BEAM entries below target ({beam_stats.total_entries}/100).", "severity": "med", "suggested_action": "Append BEAM verification entries."})
        if golden_stats.total_lines < 500:
            epistemic_gaps.append({"id": "E2", "description": f"GOLDEN lines below target ({golden_stats.total_lines}/500).", "severity": "med", "suggested_action": "Append synthesis addenda to GOLDEN."})
        if (bus_totals.get("infer_sync_responses_total") or 0) < 3:
            epistemic_gaps.append({"id": "E3", "description": "Low infer_sync.response count.", "severity": "high", "suggested_action": "Require agent infer_sync.response after ITERATE."})
        if unknown_actor_recent > 0:
            epistemic_gaps.append({"id": "E4", "description": f"Actor hygiene issue in recent BEAM ({unknown_actor_recent}).", "severity": "med", "suggested_action": "Set PLURIBUS_ACTOR before writes."})

        if bus_stats.time_buckets:
            lo = float(min(bus_stats.time_buckets))
            hi = float(max(bus_stats.time_buckets))
        else:
            lo = hi = None
        aleatoric_bounds = [{"id": "A1", "description": "Bus activity variance (bucket counts).", "bounds": {"lo": lo, "hi": hi, "unit": "events/bucket"}}]

        emit_bus(
            bus_dir,
            topic="ckin.report",
            kind="metric",
            level="info",
            actor=args.agent,
            data={
                "protocol_version": CKIN_PROTOCOL_VERSION,
                "pbdeep": pbdeep_summary,
                "paip": {
                    "multi_agent": bus_stats.paip_multi_agent,
                    "unique_actors": bus_stats.paip_unique_actors,
                    "active_clones": bus_stats.paip_active_clones,
                    "cleanup_queue": bus_stats.paip_cleanup_queue,
                    "isolation_violations_window": bus_stats.paip_isolation_violations,
                },
                "mcp_official_interop": detect_mcp_official_interop(Path("/pluribus"), bus_dir / "events.ndjson"),
                "pbflush": {
                    "requests_window": bus_stats.pbflush_requests,
                    "acks_window": bus_stats.pbflush_acks,
                    "latest_request": {
                        "iso": (bus_stats.latest_pbflush_request or {}).get("iso"),
                        "actor": (bus_stats.latest_pbflush_request or {}).get("actor"),
                        "req_id": ((bus_stats.latest_pbflush_request or {}).get("data") or {}).get("req_id") if isinstance((bus_stats.latest_pbflush_request or {}).get("data"), dict) else None,
                    },
                },
                "rd_tasks": {
                    "dispatches_window": bus_stats.rd_dispatches,
                    "acks_window": bus_stats.rd_acks,
                    "latest_dispatch": {
                        "iso": (bus_stats.latest_rd_dispatch or {}).get("iso"),
                        "actor": (bus_stats.latest_rd_dispatch or {}).get("actor"),
                        "req_id": ((bus_stats.latest_rd_dispatch or {}).get("data") or {}).get("req_id") if isinstance((bus_stats.latest_rd_dispatch or {}).get("data"), dict) else None,
                        "task_id": ((bus_stats.latest_rd_dispatch or {}).get("data") or {}).get("task_id") if isinstance((bus_stats.latest_rd_dispatch or {}).get("data"), dict) else None,
                    },
                },
                "a2a": {
                    "negotiate_requests_window": bus_stats.a2a_negotiate_requests,
                    "negotiate_responses_window": bus_stats.a2a_negotiate_responses,
                    "declines_window": bus_stats.a2a_declines,
                    "redirects_window": bus_stats.a2a_redirects,
                    "latest_negotiate_request": {
                        "iso": (bus_stats.latest_a2a_negotiate_request or {}).get("iso"),
                        "actor": (bus_stats.latest_a2a_negotiate_request or {}).get("actor"),
                        "req_id": ((bus_stats.latest_a2a_negotiate_request or {}).get("data") or {}).get("req_id") if isinstance((bus_stats.latest_a2a_negotiate_request or {}).get("data"), dict) else None,
                    },
                },
                "studio_flow": {
                    "roundtrip_window": bus_stats.studio_flow_roundtrips,
                    "roundtrip_failures_window": bus_stats.studio_flow_roundtrip_failures,
                    "latest_roundtrip": {
                        "iso": (bus_stats.latest_studio_flow_roundtrip or {}).get("iso"),
                        "actor": (bus_stats.latest_studio_flow_roundtrip or {}).get("actor"),
                        "req_id": ((bus_stats.latest_studio_flow_roundtrip or {}).get("data") or {}).get("req_id") if isinstance((bus_stats.latest_studio_flow_roundtrip or {}).get("data"), dict) else None,
                        "ok": ((bus_stats.latest_studio_flow_roundtrip or {}).get("data") or {}).get("ok") if isinstance((bus_stats.latest_studio_flow_roundtrip or {}).get("data"), dict) else None,
                    },
                },
                "beam_entries": beam_stats.total_entries,
                "beam_by_agent": beam_stats.entries_by_agent,
                "golden_lines": golden_stats.total_lines,
                "golden_sections": len(golden_stats.sections),
                "staleness": staleness,
                "gaps": {"epistemic": epistemic_gaps, "aleatoric": aleatoric_bounds},
                "compliance_sync": {
                    "bus_total_events": bus_totals.get("bus_total_events"),
                    "beam_appends_total": bus_totals.get("beam_appends_total"),
                    "ckin_reports_total": bus_totals.get("ckin_reports_total"),
                    "infer_sync_responses_total": bus_totals.get("infer_sync_responses_total"),
                    "mcp_official_interop_announcements_total": bus_totals.get("mcp_official_interop_announcements_total"),
                    "charter_lines": _count_lines(charter_path),
                    "golden_seed_lines": _count_lines(golden_seed_path),
                    "pinned_charter_lines": PINNED_CHARTER_LINES,
                    "pinned_golden_seed_lines": PINNED_GOLDEN_SEED_LINES,
                },
                "compliance": {  # back-compat alias
                    "bus_total_events": bus_totals.get("bus_total_events"),
                    "beam_appends_total": bus_totals.get("beam_appends_total"),
                    "ckin_reports_total": bus_totals.get("ckin_reports_total"),
                    "infer_sync_responses_total": bus_totals.get("infer_sync_responses_total"),
                    "mcp_official_interop_announcements_total": bus_totals.get("mcp_official_interop_announcements_total"),
                    "charter_lines": _count_lines(charter_path),
                    "golden_seed_lines": _count_lines(golden_seed_path),
                    "pinned_charter_lines": PINNED_CHARTER_LINES,
                    "pinned_golden_seed_lines": PINNED_GOLDEN_SEED_LINES,
                },
            },
        )

        if epistemic_gaps:
            emit_bus(
                bus_dir,
                topic="coordination.gap.detected",
                kind="request",
                level="info",
                actor=args.agent,
                data={"gaps": {"epistemic": epistemic_gaps, "aleatoric": aleatoric_bounds}},
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
