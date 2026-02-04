#!/usr/bin/env python3
"""
PBRESUME Operator - Recovery Orchestration for Interrupted Work (DKIN v25)

Scans ground-truth substrates for incomplete/interrupted work and emits
resumable action plans. Recovery is treated as a first-class pipeline.

Sources scanned:
1. Dialogos trace - incomplete sessions (user_prompt without matching assistant_stop)
2. Lanes state - lanes with WIP < 100%
3. Bus events - pending request/response pairs

Bus Events Emitted:
- operator.pbresume.collect - task collection started
- operator.pbresume.task - each task found
- operator.pbresume.iteration - iteration progress
- operator.pbresume.complete - all tasks done

Usage:
    python3 pbresume_operator.py --scope session|lane|all --depth 24h --dry-run
    python3 pbresume_operator.py --from-dialogos --iterative
    python3 pbresume_operator.py --help

Reference: nucleus/specs/dkin_protocol_v25_resume.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ACTIVE_TASK_STATUSES = {"in_progress", "blocked", "pending"}
PROTOCOL_VERSION = "25"

# Priority scoring constants
PRIORITY_AGE_MAX = 50  # Max points for age (>24h)
PRIORITY_AGE_24H_THRESHOLD = 86400  # 24 hours in seconds
PRIORITY_STATUS_BLOCKED = 30
PRIORITY_STATUS_IN_PROGRESS = 20
PRIORITY_STATUS_PENDING = 10
PRIORITY_LANE_BLOCKER = 25

# Session resume detection threshold
SESSION_RESUME_GAP_THRESHOLD = 300  # 5 minutes in seconds


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

def now_ts() -> float:
    """Current Unix timestamp."""
    return time.time()


def now_iso_utc() -> str:
    """Current UTC time in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def coerce_ts(value: Any) -> float:
    """Coerce timestamps to float for safe comparisons."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def parse_depth(value: str) -> float:
    """Parse depth string (e.g., '24h', '7d', '30m') to seconds."""
    raw = value.strip().lower()
    if raw.endswith("h"):
        return float(raw[:-1]) * 3600
    if raw.endswith("d"):
        return float(raw[:-1]) * 86400
    if raw.endswith("m"):
        return float(raw[:-1]) * 60
    if raw.endswith("s"):
        return float(raw[:-1])
    # Default to hours
    return float(raw) * 3600


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------

def resolve_root() -> Path:
    """Resolve Pluribus root directory."""
    root = os.environ.get("PLURIBUS_ROOT") or "/pluribus"
    return Path(root).expanduser().resolve()


def resolve_bus_dir() -> Path:
    """Resolve bus events directory."""
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip()
    if bus_dir:
        return Path(bus_dir).expanduser().resolve()
    return resolve_root() / ".pluribus" / "bus"


def resolve_dialogos_trace() -> Path:
    """Resolve dialogos trace file path."""
    trace_path = os.environ.get("PLURIBUS_DIALOGOS_TRACE", "").strip()
    if trace_path:
        return Path(trace_path).expanduser().resolve()
    return resolve_root() / ".pluribus" / "dialogos" / "trace.ndjson"


def resolve_lanes_path() -> Path:
    """Resolve lanes state file path."""
    return resolve_root() / "nucleus" / "state" / "lanes.json"


def resolve_task_ledger() -> Path:
    """Resolve task ledger file path."""
    return resolve_root() / ".pluribus" / "index" / "task_ledger.ndjson"


def default_actor() -> str:
    """Get default actor identifier."""
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pbresume"


# -----------------------------------------------------------------------------
# File I/O helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def lock_file(handle) -> None:
    """Acquire exclusive lock on file handle."""
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def unlock_file(handle) -> None:
    """Release file lock."""
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_ndjson(path: Path, obj: dict) -> None:
    """Append object as NDJSON line with file locking."""
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        lock_file(f)
        try:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        finally:
            unlock_file(f)


def read_json_safe(path: Path) -> Optional[dict]:
    """Safely read JSON file, returning None on error."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def tail_ndjson(path: Path, max_bytes: int = 2_000_000, max_lines: int = 4000) -> list[dict]:
    """
    Read last N bytes of NDJSON file efficiently.

    This avoids loading entire large files into memory.
    """
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        offset = max(0, size - max_bytes)
        with path.open("rb") as handle:
            if offset:
                handle.seek(offset, os.SEEK_SET)
            data = handle.read()
        lines = data.splitlines()
        # Skip partial first line if we started mid-file
        if offset and lines:
            lines = lines[1:]
        # Limit number of lines
        if max_lines and len(lines) > max_lines:
            lines = lines[-max_lines:]
        entries = []
        for line in lines:
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
        return entries
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Session Resume Detection
# -----------------------------------------------------------------------------

def detect_session_resume(trace_path: Path) -> bool:
    """
    Detect if this looks like a resumed session.

    Returns True if:
    1. Last session_end was >5 minutes ago
    2. There's a new session_start after that

    This indicates a gap where work may have been interrupted.
    """
    if not trace_path.exists():
        return False

    entries = tail_ndjson(trace_path, max_lines=500)
    if not entries:
        return False

    current_ts = now_ts()

    # Find the most recent session_end and session_start
    last_session_end_ts: float = 0.0
    last_session_start_ts: float = 0.0
    last_session_start_id: str = ""

    for entry in entries:
        event_type = entry.get("event_type")
        ts = coerce_ts(entry.get("ts"))

        if event_type == "session_end":
            if ts > last_session_end_ts:
                last_session_end_ts = ts
        elif event_type == "session_start":
            if ts > last_session_start_ts:
                last_session_start_ts = ts
                last_session_start_id = entry.get("session_id", "")

    # Check conditions:
    # 1. There was a session_end
    # 2. It was more than 5 minutes ago
    # 3. There's a new session_start after that
    if last_session_end_ts == 0:
        return False

    gap_since_end = current_ts - last_session_end_ts
    if gap_since_end < SESSION_RESUME_GAP_THRESHOLD:
        return False

    # Check if there's a session_start after the session_end
    if last_session_start_ts > last_session_end_ts:
        return True

    return False


def get_session_gap_info(trace_path: Path) -> dict:
    """
    Get detailed information about the session gap.

    Returns dict with:
    - gap_seconds: time between last session_end and now (or new session_start)
    - last_session_id: ID of the last ended session
    - new_session_id: ID of the new session (if any)
    - is_resume: whether this looks like a resume
    """
    if not trace_path.exists():
        return {"is_resume": False, "gap_seconds": 0}

    entries = tail_ndjson(trace_path, max_lines=500)
    if not entries:
        return {"is_resume": False, "gap_seconds": 0}

    current_ts = now_ts()

    last_session_end_ts: float = 0.0
    last_session_end_id: str = ""
    last_session_start_ts: float = 0.0
    last_session_start_id: str = ""

    for entry in entries:
        event_type = entry.get("event_type")
        ts = coerce_ts(entry.get("ts"))
        session_id = entry.get("session_id", "")

        if event_type == "session_end":
            if ts > last_session_end_ts:
                last_session_end_ts = ts
                last_session_end_id = session_id
        elif event_type == "session_start":
            if ts > last_session_start_ts:
                last_session_start_ts = ts
                last_session_start_id = session_id

    if last_session_end_ts == 0:
        return {"is_resume": False, "gap_seconds": 0}

    # Calculate gap
    if last_session_start_ts > last_session_end_ts:
        gap_seconds = last_session_start_ts - last_session_end_ts
        is_resume = gap_seconds >= SESSION_RESUME_GAP_THRESHOLD
    else:
        gap_seconds = current_ts - last_session_end_ts
        is_resume = False

    return {
        "is_resume": is_resume,
        "gap_seconds": gap_seconds,
        "gap_formatted": format_duration(gap_seconds),
        "last_session_id": last_session_end_id,
        "new_session_id": last_session_start_id if last_session_start_ts > last_session_end_ts else "",
    }


# -----------------------------------------------------------------------------
# Priority Scoring
# -----------------------------------------------------------------------------

def calculate_task_priority(task: dict) -> int:
    """
    Calculate priority score for a task.

    Scoring:
    - Age-based: older = higher priority (max 50 points for >24h)
    - Status-based: blocked = +30, in_progress = +20, pending = +10
    """
    priority = 0

    # Age-based scoring
    ts = coerce_ts(task.get("ts"))
    if ts > 0:
        age_seconds = now_ts() - ts
        # Scale from 0 to PRIORITY_AGE_MAX over 24 hours
        age_ratio = min(1.0, age_seconds / PRIORITY_AGE_24H_THRESHOLD)
        priority += int(age_ratio * PRIORITY_AGE_MAX)

    # Status-based scoring
    status = task.get("status", "pending")
    if status == "blocked":
        priority += PRIORITY_STATUS_BLOCKED
    elif status == "in_progress":
        priority += PRIORITY_STATUS_IN_PROGRESS
    elif status == "pending":
        priority += PRIORITY_STATUS_PENDING

    return priority


def calculate_dialogos_priority(prompt: dict) -> int:
    """Calculate priority score for a dialogos prompt."""
    priority = 0

    # Age-based scoring
    ts = coerce_ts(prompt.get("ts"))
    if ts > 0:
        age_seconds = now_ts() - ts
        age_ratio = min(1.0, age_seconds / PRIORITY_AGE_24H_THRESHOLD)
        priority += int(age_ratio * PRIORITY_AGE_MAX)

    # Dialogos prompts default to pending priority
    priority += PRIORITY_STATUS_PENDING

    return priority


def calculate_lane_priority(lane: dict) -> int:
    """
    Calculate priority score for a lane.

    Scoring:
    - Age-based: not applicable (lanes don't have timestamps)
    - Status-based: blocked lanes get +30
    - Blocker bonus: lanes with blockers get +25
    """
    priority = 0

    # Status-based scoring
    status = lane.get("status", "")
    if status == "blocked":
        priority += PRIORITY_STATUS_BLOCKED
    elif status == "red":
        priority += PRIORITY_STATUS_BLOCKED
    elif status == "yellow":
        priority += PRIORITY_STATUS_IN_PROGRESS
    else:
        priority += PRIORITY_STATUS_PENDING

    # Blocker bonus
    blockers = lane.get("blockers", [])
    if blockers:
        priority += PRIORITY_LANE_BLOCKER

    # WIP factor: lower WIP = higher priority
    wip_pct = lane.get("wip_pct", 0)
    remaining_pct = 100 - wip_pct
    # Scale remaining from 0-100 to 0-20 points
    priority += int(remaining_pct / 5)

    return priority


def calculate_pending_request_priority(req: dict) -> int:
    """Calculate priority score for a pending request."""
    priority = 0

    # Age-based scoring
    ts = coerce_ts(req.get("ts"))
    if ts > 0:
        age_seconds = now_ts() - ts
        age_ratio = min(1.0, age_seconds / PRIORITY_AGE_24H_THRESHOLD)
        priority += int(age_ratio * PRIORITY_AGE_MAX)

    # Pending requests are in_progress status
    priority += PRIORITY_STATUS_IN_PROGRESS

    return priority


def add_priority_to_items(report_data: dict) -> dict:
    """Add priority scores to all items in the report and sort by priority."""
    # Tasks
    for task in report_data.get("open_tasks", []):
        task["priority"] = calculate_task_priority(task)
    report_data["open_tasks"] = sorted(
        report_data.get("open_tasks", []),
        key=lambda x: x.get("priority", 0),
        reverse=True
    )

    # Dialogos
    for prompt in report_data.get("dialogos_pending", []):
        prompt["priority"] = calculate_dialogos_priority(prompt)
    report_data["dialogos_pending"] = sorted(
        report_data.get("dialogos_pending", []),
        key=lambda x: x.get("priority", 0),
        reverse=True
    )

    # Lanes
    for lane in report_data.get("incomplete_lanes", []):
        lane["priority"] = calculate_lane_priority(lane)
    report_data["incomplete_lanes"] = sorted(
        report_data.get("incomplete_lanes", []),
        key=lambda x: x.get("priority", 0),
        reverse=True
    )

    # Pending requests
    for req in report_data.get("pending_requests", []):
        req["priority"] = calculate_pending_request_priority(req)
    report_data["pending_requests"] = sorted(
        report_data.get("pending_requests", []),
        key=lambda x: x.get("priority", 0),
        reverse=True
    )

    return report_data


# -----------------------------------------------------------------------------
# Summary Generation
# -----------------------------------------------------------------------------

def generate_summary(report_data: dict) -> str:
    """
    Generate a concise natural language summary of the recovery report.

    Example output:
    "Found 3 incomplete items: 1 blocked lane (dashboard-fix),
     2 pending dialogos sessions from 2h ago. Recommend: resume dashboard-fix first."
    """
    counts = report_data.get("counts", {})
    total = counts.get("total", 0)

    if total == 0:
        return "No incomplete items found. All work appears to be complete."

    parts = []

    # Count items by type
    lanes = report_data.get("incomplete_lanes", [])
    dialogos = report_data.get("dialogos_pending", [])
    tasks = report_data.get("open_tasks", [])
    requests = report_data.get("pending_requests", [])

    # Blocked lanes
    blocked_lanes = [l for l in lanes if l.get("blockers") or l.get("status") == "blocked"]
    if blocked_lanes:
        names = [l.get("name", l.get("id", "unknown")) for l in blocked_lanes[:2]]
        names_str = ", ".join(names)
        if len(blocked_lanes) > 2:
            names_str += f" (+{len(blocked_lanes) - 2} more)"
        parts.append(f"{len(blocked_lanes)} blocked lane{'s' if len(blocked_lanes) > 1 else ''} ({names_str})")

    # Incomplete lanes (not blocked)
    other_lanes = [l for l in lanes if l not in blocked_lanes]
    if other_lanes:
        parts.append(f"{len(other_lanes)} incomplete lane{'s' if len(other_lanes) > 1 else ''}")

    # Dialogos sessions
    if dialogos:
        # Get age of oldest
        oldest_age = max((d.get("age", "?") for d in dialogos), default="?")
        parts.append(f"{len(dialogos)} pending dialogos session{'s' if len(dialogos) > 1 else ''} from {oldest_age} ago")

    # Open tasks
    if tasks:
        blocked_tasks = [t for t in tasks if t.get("status") == "blocked"]
        in_progress_tasks = [t for t in tasks if t.get("status") == "in_progress"]
        pending_tasks = [t for t in tasks if t.get("status") == "pending"]

        task_parts = []
        if blocked_tasks:
            task_parts.append(f"{len(blocked_tasks)} blocked")
        if in_progress_tasks:
            task_parts.append(f"{len(in_progress_tasks)} in-progress")
        if pending_tasks:
            task_parts.append(f"{len(pending_tasks)} pending")

        if task_parts:
            parts.append(f"{len(tasks)} task{'s' if len(tasks) > 1 else ''} ({', '.join(task_parts)})")

    # Pending requests
    if requests:
        parts.append(f"{len(requests)} pending bus request{'s' if len(requests) > 1 else ''}")

    # Build the found section
    summary = f"Found {total} incomplete item{'s' if total > 1 else ''}"
    if parts:
        summary += ": " + ", ".join(parts)
    summary += "."

    # Add recommendation
    recommendation = generate_recommendation(report_data)
    if recommendation:
        summary += f" Recommend: {recommendation}"

    return summary


def generate_recommendation(report_data: dict) -> str:
    """Generate a recommendation for what to resume first."""
    # Collect all items with priorities
    all_items = []

    for lane in report_data.get("incomplete_lanes", []):
        all_items.append({
            "type": "lane",
            "name": lane.get("name", lane.get("id", "unknown")),
            "priority": lane.get("priority", 0),
            "blocked": bool(lane.get("blockers")),
        })

    for task in report_data.get("open_tasks", []):
        all_items.append({
            "type": "task",
            "name": task.get("desc") or task.get("run_id", "unknown"),
            "priority": task.get("priority", 0),
            "blocked": task.get("status") == "blocked",
        })

    for prompt in report_data.get("dialogos_pending", []):
        all_items.append({
            "type": "dialogos",
            "name": prompt.get("summary") or prompt.get("session_id", "unknown"),
            "priority": prompt.get("priority", 0),
            "blocked": False,
        })

    if not all_items:
        return ""

    # Sort by priority
    all_items.sort(key=lambda x: x["priority"], reverse=True)
    top_item = all_items[0]

    # Truncate name for readability
    name = top_item["name"]
    if len(name) > 40:
        name = name[:37] + "..."

    if top_item["type"] == "lane":
        if top_item["blocked"]:
            return f"unblock lane '{name}' first."
        return f"resume lane '{name}' first."
    elif top_item["type"] == "task":
        if top_item["blocked"]:
            return f"unblock task '{name}' first."
        return f"resume task '{name}' first."
    elif top_item["type"] == "dialogos":
        return f"continue dialogos session '{name}'."

    return f"resume '{name}' first."


# -----------------------------------------------------------------------------
# Bus event emission
# -----------------------------------------------------------------------------

def emit_bus(
    bus_dir: Path,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict,
    trace_id: Optional[str] = None,
) -> str:
    """Emit event to Pluribus bus with atomic file locking."""
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": now_ts(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    if trace_id:
        evt["trace_id"] = trace_id
    elif os.environ.get("PLURIBUS_TRACE_ID"):
        evt["trace_id"] = os.environ.get("PLURIBUS_TRACE_ID")

    events_path = bus_dir / "events.ndjson"
    append_ndjson(events_path, evt)
    return evt_id


# -----------------------------------------------------------------------------
# Dialogos trace helpers
# -----------------------------------------------------------------------------

def append_dialogos_trace(trace_path: Path, event: dict) -> None:
    """Append event to dialogos trace for ground truth recovery."""
    event.setdefault("id", str(uuid.uuid4()))
    event.setdefault("ts", now_ts())
    event.setdefault("iso", now_iso_utc())
    append_ndjson(trace_path, event)


def scan_dialogos_incomplete(trace_path: Path, since_ts: float) -> list[dict]:
    """
    Scan dialogos trace for incomplete sessions.

    A session is incomplete if there's a user_prompt without a matching
    assistant_stop event afterwards.
    """
    entries = tail_ndjson(trace_path)

    # Track last prompt and stop per session
    last_prompt: dict[str, dict] = {}
    last_stop: dict[str, float] = {}

    for entry in entries:
        ts = coerce_ts(entry.get("ts"))
        if ts < since_ts:
            continue

        session_id = entry.get("session_id") or "unknown"
        event_type = entry.get("event_type")

        if event_type == "user_prompt":
            last_prompt[session_id] = entry
        elif event_type in {"assistant_stop", "session_end"}:
            last_stop[session_id] = ts

    # Find prompts without matching stops
    pending = []
    for session_id, prompt in last_prompt.items():
        prompt_ts = coerce_ts(prompt.get("ts"))
        stop_ts = last_stop.get(session_id, 0)
        if stop_ts < prompt_ts:
            pending.append(prompt)

    return pending


# -----------------------------------------------------------------------------
# Lane state scanning
# -----------------------------------------------------------------------------

@dataclass
class LaneStatus:
    """Represents the status of a work lane."""
    id: str
    name: str
    status: str
    wip_pct: int
    owner: str
    description: str
    blockers: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)


def scan_incomplete_lanes(lanes_path: Path) -> list[LaneStatus]:
    """Scan lanes.json for lanes with WIP < 100%."""
    data = read_json_safe(lanes_path)
    if not data:
        return []

    lanes = data.get("lanes", [])
    incomplete = []

    for lane in lanes:
        wip_pct = lane.get("wip_pct", 0)
        if wip_pct < 100:
            incomplete.append(LaneStatus(
                id=lane.get("id", "unknown"),
                name=lane.get("name", ""),
                status=lane.get("status", "unknown"),
                wip_pct=wip_pct,
                owner=lane.get("owner", ""),
                description=lane.get("description", ""),
                blockers=lane.get("blockers", []),
                next_actions=lane.get("next_actions", []),
            ))

    return incomplete


# -----------------------------------------------------------------------------
# Task ledger scanning
# -----------------------------------------------------------------------------

def scan_open_tasks(ledger_path: Path, since_ts: float) -> list[dict]:
    """Scan task ledger for open/in-progress tasks."""
    entries = tail_ndjson(ledger_path)

    # Group by run_id, keep latest entry per task
    latest: dict[str, dict] = {}
    for entry in entries:
        ts = coerce_ts(entry.get("ts"))
        if ts < since_ts:
            continue

        run_id = entry.get("run_id") or entry.get("id") or entry.get("req_id")
        if not run_id:
            continue
        latest[run_id] = entry

    # Filter to active statuses
    tasks = []
    for entry in latest.values():
        status = entry.get("status")
        if status in ACTIVE_TASK_STATUSES:
            tasks.append(entry)

    return tasks


# -----------------------------------------------------------------------------
# Bus event scanning (pending request/response pairs)
# -----------------------------------------------------------------------------

def scan_pending_requests(bus_dir: Path, since_ts: float, timeout_s: float = 300.0) -> list[dict]:
    """
    Scan bus events for request events without matching responses.

    A request is considered pending if:
    1. It was emitted after since_ts
    2. No response with matching req_id was found
    3. The request is older than timeout_s (to avoid flagging in-flight requests)
    """
    events_path = bus_dir / "events.ndjson"
    entries = tail_ndjson(events_path)

    current_ts = now_ts()
    cutoff_ts = current_ts - timeout_s

    # Track requests and responses by req_id
    requests: dict[str, dict] = {}
    responded: set[str] = set()

    for entry in entries:
        ts = coerce_ts(entry.get("ts"))
        if ts < since_ts:
            continue

        kind = entry.get("kind")
        data = entry.get("data", {})
        req_id = data.get("req_id") or data.get("request_id") or entry.get("req_id")

        if not req_id:
            continue

        if kind == "request":
            requests[req_id] = entry
        elif kind == "response":
            responded.add(req_id)

    # Find pending requests (old enough to be considered stale)
    pending = []
    for req_id, entry in requests.items():
        if req_id in responded:
            continue
        ts = coerce_ts(entry.get("ts"))
        if ts < cutoff_ts:  # Only flag if old enough
            pending.append(entry)

    return pending


# -----------------------------------------------------------------------------
# Apology frame
# -----------------------------------------------------------------------------

APOLOGY_FRAME = """
Apologies for the interruption. I am collecting incomplete work from ground-truth
traces and resuming operations. Here is the recovery report:
""".strip()


def apology_frame() -> str:
    """Generate apology acknowledgment for interrupted sessions."""
    return APOLOGY_FRAME


# -----------------------------------------------------------------------------
# Task summarization
# -----------------------------------------------------------------------------

def summarize_task(entry: dict) -> dict:
    """Summarize a task entry for the recovery report."""
    meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
    data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
    ts = coerce_ts(entry.get("ts"))
    return {
        "run_id": entry.get("run_id"),
        "req_id": entry.get("req_id") or data.get("req_id"),
        "actor": entry.get("actor"),
        "status": entry.get("status"),
        "topic": entry.get("topic"),
        "desc": meta.get("desc") or meta.get("step") or data.get("intent"),
        "ts": entry.get("ts"),
        "age": format_duration(now_ts() - (ts or now_ts())),
    }


def summarize_dialogos_prompt(prompt: dict) -> dict:
    """Summarize a dialogos prompt for the recovery report."""
    ts = coerce_ts(prompt.get("ts"))
    return {
        "id": prompt.get("id"),
        "req_id": prompt.get("req_id"),
        "session_id": prompt.get("session_id"),
        "summary": prompt.get("summary") or prompt.get("prompt_preview"),
        "prompt_sha256": prompt.get("prompt_sha256"),
        "ts": prompt.get("ts"),
        "age": format_duration(now_ts() - (ts or now_ts())),
    }


def summarize_lane(lane: LaneStatus) -> dict:
    """Summarize a lane for the recovery report."""
    return {
        "id": lane.id,
        "name": lane.name,
        "status": lane.status,
        "wip_pct": lane.wip_pct,
        "owner": lane.owner,
        "blockers": lane.blockers,
        "next_actions": lane.next_actions,
        "remaining_pct": 100 - lane.wip_pct,
    }


def summarize_pending_request(entry: dict) -> dict:
    """Summarize a pending bus request."""
    data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
    ts = coerce_ts(entry.get("ts"))
    return {
        "id": entry.get("id"),
        "req_id": data.get("req_id") or data.get("request_id"),
        "topic": entry.get("topic"),
        "actor": entry.get("actor"),
        "ts": entry.get("ts"),
        "age": format_duration(now_ts() - (ts or now_ts())),
    }


# -----------------------------------------------------------------------------
# Recovery report generation
# -----------------------------------------------------------------------------

@dataclass
class RecoveryReport:
    """Complete recovery report with all pending work."""
    req_id: str
    scope: str
    since_ts: float
    depth_str: str
    actor: str
    session_id: Optional[str]
    lane: Optional[str]
    apology: str
    open_tasks: list[dict]
    dialogos_pending: list[dict]
    incomplete_lanes: list[dict]
    pending_requests: list[dict]

    @property
    def counts(self) -> dict:
        return {
            "tasks": len(self.open_tasks),
            "dialogos_pending": len(self.dialogos_pending),
            "incomplete_lanes": len(self.incomplete_lanes),
            "pending_requests": len(self.pending_requests),
            "total": (
                len(self.open_tasks) +
                len(self.dialogos_pending) +
                len(self.incomplete_lanes) +
                len(self.pending_requests)
            ),
        }

    def to_dict(self) -> dict:
        return {
            "req_id": self.req_id,
            "scope": self.scope,
            "since_ts": self.since_ts,
            "depth": self.depth_str,
            "actor": self.actor,
            "session_id": self.session_id,
            "lane": self.lane,
            "apology": self.apology,
            "open_tasks": self.open_tasks,
            "dialogos_pending": self.dialogos_pending,
            "incomplete_lanes": self.incomplete_lanes,
            "pending_requests": self.pending_requests,
            "counts": self.counts,
            "protocol_version": PROTOCOL_VERSION,
            "ts": now_ts(),
            "iso": now_iso_utc(),
        }


def collect_recovery_data(
    scope: str,
    depth_s: float,
    depth_str: str,
    from_dialogos: bool,
    from_lanes: bool,
    from_bus: bool,
) -> RecoveryReport:
    """Collect all pending/incomplete work from ground-truth sources."""
    now = now_ts()
    since_ts = now - depth_s

    req_id = f"pbresume-{uuid.uuid4().hex[:12]}"
    actor = default_actor()
    session_id = os.environ.get("PLURIBUS_SESSION_ID")
    lane = os.environ.get("PLURIBUS_LANE")

    # Scan sources based on scope
    open_tasks: list[dict] = []
    dialogos_pending: list[dict] = []
    incomplete_lanes: list[dict] = []
    pending_requests: list[dict] = []

    # Task ledger (always scan for 'all' or 'session' scope)
    if scope in ("all", "session"):
        ledger_path = resolve_task_ledger()
        raw_tasks = scan_open_tasks(ledger_path, since_ts)
        open_tasks = [summarize_task(t) for t in raw_tasks]

    # Dialogos trace
    if from_dialogos or scope in ("all", "session"):
        trace_path = resolve_dialogos_trace()
        raw_prompts = scan_dialogos_incomplete(trace_path, since_ts)
        dialogos_pending = [summarize_dialogos_prompt(p) for p in raw_prompts]

    # Lanes state
    if from_lanes or scope in ("all", "lane"):
        lanes_path = resolve_lanes_path()
        raw_lanes = scan_incomplete_lanes(lanes_path)
        incomplete_lanes = [summarize_lane(l) for l in raw_lanes]

    # Bus events (pending requests)
    if from_bus or scope == "all":
        bus_dir = resolve_bus_dir()
        raw_requests = scan_pending_requests(bus_dir, since_ts)
        pending_requests = [summarize_pending_request(r) for r in raw_requests]

    return RecoveryReport(
        req_id=req_id,
        scope=scope,
        since_ts=since_ts,
        depth_str=depth_str,
        actor=actor,
        session_id=session_id,
        lane=lane,
        apology=apology_frame(),
        open_tasks=open_tasks,
        dialogos_pending=dialogos_pending,
        incomplete_lanes=incomplete_lanes,
        pending_requests=pending_requests,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    p = argparse.ArgumentParser(
        prog="pbresume_operator.py",
        description="PBRESUME recovery operator - scan for incomplete work (DKIN v25)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all sources, last 24 hours
  python3 pbresume_operator.py --scope all --depth 24h

  # Dry run - don't emit iteration events
  python3 pbresume_operator.py --scope session --dry-run

  # Scan dialogos trace specifically
  python3 pbresume_operator.py --from-dialogos --iterative

  # Scan lanes for WIP < 100%
  python3 pbresume_operator.py --scope lane

  # JSON output only (for piping)
  python3 pbresume_operator.py --quiet
""",
    )
    p.add_argument(
        "--scope",
        default="all",
        choices=["session", "lane", "all"],
        help="Scope of recovery scan (default: all)",
    )
    p.add_argument(
        "--depth",
        default="24h",
        help="How far back to scan (e.g., 24h, 7d, 30m) (default: 24h)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect data but don't emit iteration events",
    )
    p.add_argument(
        "--iterative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit per-task iteration events (default: true)",
    )
    p.add_argument(
        "--from-dialogos",
        action="store_true",
        help="Explicitly include dialogos trace scan",
    )
    p.add_argument(
        "--from-lanes",
        action="store_true",
        help="Explicitly include lanes state scan",
    )
    p.add_argument(
        "--from-bus",
        action="store_true",
        help="Explicitly include bus event scan",
    )
    p.add_argument(
        "--emit-apology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include apology frame in output (default: true)",
    )
    p.add_argument(
        "--write-trace",
        action="store_true",
        help="Write recovery event to dialogos trace for ground truth",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode - JSON output only, no status messages",
    )
    p.add_argument(
        "--format",
        default="json",
        choices=["json", "markdown", "text"],
        help="Output format (default: json)",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Auto-trigger mode: detect resume and run automatically if detected",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Generate concise natural language summary",
    )
    p.add_argument(
        "--priority",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add priority scores and sort by priority (default: true)",
    )
    return p


def format_markdown(report: RecoveryReport) -> str:
    """Format report as markdown."""
    lines = [
        f"# PBRESUME Recovery Report",
        "",
        f"**Request ID:** `{report.req_id}`",
        f"**Scope:** {report.scope}",
        f"**Depth:** {report.depth_str}",
        f"**Actor:** {report.actor}",
        f"**Timestamp:** {now_iso_utc()}",
        "",
    ]

    if report.apology:
        lines.extend([
            "## Apology Frame",
            "",
            f"> {report.apology}",
            "",
        ])

    lines.extend([
        "## Summary",
        "",
        f"| Category | Count |",
        f"|----------|-------|",
        f"| Open Tasks | {report.counts['tasks']} |",
        f"| Dialogos Pending | {report.counts['dialogos_pending']} |",
        f"| Incomplete Lanes | {report.counts['incomplete_lanes']} |",
        f"| Pending Requests | {report.counts['pending_requests']} |",
        f"| **Total** | **{report.counts['total']}** |",
        "",
    ])

    if report.incomplete_lanes:
        lines.extend([
            "## Incomplete Lanes",
            "",
        ])
        for lane in report.incomplete_lanes:
            wip = lane.get("wip_pct", 0)
            remaining = 100 - wip
            bar_full = int(wip / 10)
            bar_empty = 10 - bar_full
            bar = "█" * bar_full + "░" * bar_empty
            lines.append(f"- **{lane.get('name')}** [{bar}] {wip}% ({remaining}% remaining)")
            if lane.get("blockers"):
                lines.append(f"  - Blockers: {', '.join(lane['blockers'])}")
            if lane.get("next_actions"):
                for action in lane["next_actions"][:2]:
                    lines.append(f"  - Next: {action}")
        lines.append("")

    if report.dialogos_pending:
        lines.extend([
            "## Pending Dialogos Sessions",
            "",
        ])
        for prompt in report.dialogos_pending:
            age = prompt.get("age", "?")
            summary = prompt.get("summary") or "(no summary)"
            lines.append(f"- `{prompt.get('session_id', '?')}` ({age} ago): {summary[:60]}...")
        lines.append("")

    if report.open_tasks:
        lines.extend([
            "## Open Tasks",
            "",
        ])
        for task in report.open_tasks:
            age = task.get("age", "?")
            desc = task.get("desc") or task.get("topic") or "(no description)"
            lines.append(f"- `{task.get('run_id', '?')}` [{task.get('status', '?')}] ({age} ago): {desc}")
        lines.append("")

    if report.pending_requests:
        lines.extend([
            "## Pending Bus Requests",
            "",
        ])
        for req in report.pending_requests:
            age = req.get("age", "?")
            topic = req.get("topic") or "(no topic)"
            lines.append(f"- `{req.get('req_id', '?')}` {topic} ({age} ago)")
        lines.append("")

    return "\n".join(lines)


def format_text(report: RecoveryReport) -> str:
    """Format report as plain text."""
    lines = [
        "=" * 60,
        "PBRESUME RECOVERY REPORT",
        "=" * 60,
        "",
        f"Request ID: {report.req_id}",
        f"Scope: {report.scope}",
        f"Depth: {report.depth_str}",
        f"Actor: {report.actor}",
        f"Timestamp: {now_iso_utc()}",
        "",
    ]

    if report.apology:
        lines.extend([
            "-" * 40,
            report.apology,
            "-" * 40,
            "",
        ])

    lines.extend([
        "SUMMARY:",
        f"  Open Tasks:        {report.counts['tasks']}",
        f"  Dialogos Pending:  {report.counts['dialogos_pending']}",
        f"  Incomplete Lanes:  {report.counts['incomplete_lanes']}",
        f"  Pending Requests:  {report.counts['pending_requests']}",
        f"  Total:             {report.counts['total']}",
        "",
    ])

    if report.incomplete_lanes:
        lines.append("INCOMPLETE LANES:")
        for lane in report.incomplete_lanes:
            lines.append(f"  [{lane.get('wip_pct', 0):3d}%] {lane.get('name')} - {lane.get('owner')}")
        lines.append("")

    if report.dialogos_pending:
        lines.append("PENDING DIALOGOS:")
        for prompt in report.dialogos_pending[:5]:
            summary = prompt.get("summary") or "(no summary)"
            lines.append(f"  {prompt.get('session_id', '?')}: {summary[:40]}...")
        if len(report.dialogos_pending) > 5:
            lines.append(f"  ... and {len(report.dialogos_pending) - 5} more")
        lines.append("")

    if report.open_tasks:
        lines.append("OPEN TASKS:")
        for task in report.open_tasks[:5]:
            desc = task.get("desc") or task.get("topic") or "(no description)"
            lines.append(f"  {task.get('run_id', '?')}: {desc[:40]}...")
        if len(report.open_tasks) > 5:
            lines.append(f"  ... and {len(report.open_tasks) - 5} more")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def run_auto_resume(quiet: bool = False) -> int:
    """
    Run auto-resume detection and trigger.

    Returns:
        0 if no resume needed or resume completed successfully
        1 if error occurred
    """
    trace_path = resolve_dialogos_trace()
    bus_dir = resolve_bus_dir()
    ensure_dir(bus_dir)

    actor = default_actor()

    # Check if this looks like a resumed session
    if not detect_session_resume(trace_path):
        if not quiet:
            print("[pbresume:auto] No session resume detected", file=sys.stderr)
        return 0

    # Get session gap info for logging
    gap_info = get_session_gap_info(trace_path)

    if not quiet:
        print(f"[pbresume:auto] Session resume detected (gap: {gap_info.get('gap_formatted', '?')})", file=sys.stderr)

    # Emit auto-triggered event
    emit_bus(
        bus_dir,
        topic="operator.pbresume.auto_triggered",
        kind="request",
        level="info",
        actor=actor,
        data={
            "trigger": "session_resume",
            "gap_seconds": gap_info.get("gap_seconds", 0),
            "gap_formatted": gap_info.get("gap_formatted", "?"),
            "last_session_id": gap_info.get("last_session_id", ""),
            "new_session_id": gap_info.get("new_session_id", ""),
        },
    )

    # Run pbresume with session scope
    return main(["--scope", "session", "--quiet" if quiet else "--format", "text"])


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    args = build_parser().parse_args(argv)

    # Handle auto mode first
    if args.auto:
        return run_auto_resume(quiet=args.quiet)

    # Parse depth
    try:
        depth_s = parse_depth(args.depth)
    except ValueError:
        print(f"Error: Invalid depth format: {args.depth}", file=sys.stderr)
        return 1

    bus_dir = resolve_bus_dir()
    ensure_dir(bus_dir)

    # Collect recovery data
    report = collect_recovery_data(
        scope=args.scope,
        depth_s=depth_s,
        depth_str=args.depth,
        from_dialogos=args.from_dialogos,
        from_lanes=args.from_lanes,
        from_bus=args.from_bus,
    )

    # Remove apology if not requested
    if not args.emit_apology:
        report.apology = ""

    actor = report.actor
    report_data = report.to_dict()

    # Add priority scoring if enabled
    if args.priority:
        report_data = add_priority_to_items(report_data)

    # Generate summary if requested
    summary_text = ""
    if args.summary:
        summary_text = generate_summary(report_data)
        report_data["summary"] = summary_text

    # Emit collect event
    emit_bus(
        bus_dir,
        topic="operator.pbresume.collect",
        kind="request",
        level="info",
        actor=actor,
        data=report_data,
    )

    # Emit per-task iteration events if not dry-run
    if not args.dry_run and args.iterative:
        # Tasks
        for task in report.open_tasks:
            emit_bus(
                bus_dir,
                topic="operator.pbresume.task",
                kind="request",
                level="info",
                actor=actor,
                data={"req_id": report.req_id, "task": task},
            )
            emit_bus(
                bus_dir,
                topic="operator.pbresume.iteration",
                kind="request",
                level="info",
                actor=actor,
                data={"req_id": report.req_id, "type": "task", "item": task},
            )

        # Dialogos prompts
        for prompt in report.dialogos_pending:
            emit_bus(
                bus_dir,
                topic="dialogos.resume",
                kind="request",
                level="info",
                actor=actor,
                data={"req_id": report.req_id, "prompt": prompt},
            )
            emit_bus(
                bus_dir,
                topic="operator.pbresume.iteration",
                kind="request",
                level="info",
                actor=actor,
                data={"req_id": report.req_id, "type": "dialogos", "item": prompt},
            )

        # Lanes
        for lane in report.incomplete_lanes:
            emit_bus(
                bus_dir,
                topic="operator.pbresume.iteration",
                kind="request",
                level="info",
                actor=actor,
                data={"req_id": report.req_id, "type": "lane", "item": lane},
            )

        # Pending requests
        for req in report.pending_requests:
            emit_bus(
                bus_dir,
                topic="operator.pbresume.iteration",
                kind="request",
                level="info",
                actor=actor,
                data={"req_id": report.req_id, "type": "pending_request", "item": req},
            )

    # Write to dialogos trace if requested
    if args.write_trace:
        trace_path = resolve_dialogos_trace()
        append_dialogos_trace(trace_path, {
            "event_type": "pbresume_recovery",
            "req_id": report.req_id,
            "scope": report.scope,
            "counts": report.counts,
            "actor": actor,
        })

    # Emit completion event
    emit_bus(
        bus_dir,
        topic="operator.pbresume.complete",
        kind="response",
        level="info",
        actor=actor,
        data={
            "req_id": report.req_id,
            "counts": report.counts,
            "status": "success" if report.counts["total"] == 0 else "pending",
        },
    )

    # Output
    if args.summary and not args.quiet:
        # Summary mode - just print the summary
        print(summary_text)
    elif args.format == "json":
        print(json.dumps(report_data, indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        print(format_markdown(report))
    else:
        print(format_text(report))

    if not args.quiet and not args.summary:
        print(f"\n[pbresume] Collected {report.counts['total']} pending items", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
