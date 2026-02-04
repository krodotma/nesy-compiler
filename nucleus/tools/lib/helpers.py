#!/usr/bin/env python3
"""
Pluribus Shared Helpers Library
================================

Consolidated utility functions for all nucleus tools.
Eliminates code duplication across the toolchain.

Usage:
    from nucleus.tools.lib.helpers import now_iso_utc, ensure_dir, append_ndjson

    # Timestamp
    ts = now_iso_utc()  # "2025-12-31T00:00:00Z"

    # Directory creation
    ensure_dir(Path("/pluribus/data/output"))

    # NDJSON append
    append_ndjson(Path("/path/to/file.ndjson"), {"key": "value"})

    # Bus event emission (convenience wrapper)
    emit_bus_event("topic.name", {"data": 123}, actor="my-tool")

DKIN v28 Compliant - Shared utilities for citizen agents.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

sys.dont_write_bytecode = True

# ---------------------------------------------------------------------------
# Timestamp Utilities
# ---------------------------------------------------------------------------

def now_iso_utc() -> str:
    """
    Return current UTC time in ISO 8601 format.

    Returns:
        str: Timestamp like "2025-12-31T12:34:56Z"

    Example:
        >>> ts = now_iso_utc()
        >>> print(ts)  # "2025-12-31T12:34:56Z"
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_epoch_ms() -> int:
    """
    Return current UTC time as milliseconds since epoch.

    Returns:
        int: Milliseconds since Unix epoch
    """
    return int(time.time() * 1000)


def parse_iso_utc(iso_str: str) -> float:
    """
    Parse ISO 8601 UTC timestamp to Unix epoch seconds.

    Args:
        iso_str: ISO 8601 string (e.g., "2025-12-31T12:34:56Z")

    Returns:
        float: Unix epoch seconds

    Raises:
        ValueError: If string is not valid ISO 8601 format
    """
    # Remove trailing Z and parse
    clean = iso_str.rstrip("Z")
    try:
        struct = time.strptime(clean, "%Y-%m-%dT%H:%M:%S")
        return time.mktime(struct) - time.timezone
    except ValueError:
        # Try with milliseconds
        struct = time.strptime(clean.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        return time.mktime(struct) - time.timezone


# ---------------------------------------------------------------------------
# Directory/File Utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[Path, str]) -> Path:
    """
    Create directory if it doesn't exist (including parents).

    Args:
        path: Directory path to ensure exists

    Returns:
        Path: The path as a Path object

    Example:
        >>> ensure_dir("/pluribus/data/output")
        PosixPath('/pluribus/data/output')
    """
    p = Path(path) if isinstance(path, str) else path
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent_dir(file_path: Union[Path, str]) -> Path:
    """
    Ensure parent directory of a file path exists.

    Args:
        file_path: Path to a file (parent directory will be created)

    Returns:
        Path: The file path as a Path object
    """
    p = Path(file_path) if isinstance(file_path, str) else file_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# NDJSON Utilities (Append-Only Pattern)
# ---------------------------------------------------------------------------

def append_ndjson(
    path: Union[Path, str],
    data: Dict[str, Any],
    ensure_parent: bool = True,
    atomic: bool = True
) -> None:
    """
    Append a JSON object as a new line to an NDJSON file.

    Uses file locking for atomic writes to prevent interleaving
    in concurrent scenarios (multiple agents writing).

    Args:
        path: Path to the NDJSON file
        data: Dictionary to serialize and append
        ensure_parent: Create parent directories if missing
        atomic: Use file locking for thread safety

    Example:
        >>> append_ndjson("/path/to/events.ndjson", {"event": "test", "ts": 123})
    """
    p = Path(path) if isinstance(path, str) else path

    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"

    if atomic:
        with p.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    else:
        with p.open("a", encoding="utf-8") as f:
            f.write(line)


def read_ndjson(path: Union[Path, str]) -> list[Dict[str, Any]]:
    """
    Read all records from an NDJSON file.

    Args:
        path: Path to the NDJSON file

    Returns:
        list: List of parsed JSON objects

    Note:
        Skips empty lines and invalid JSON lines (logs to stderr).
    """
    p = Path(path) if isinstance(path, str) else path

    if not p.exists():
        return []

    results = []
    with p.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[helpers] Invalid JSON at line {line_num}: {e}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Bus Event Emission (Convenience Wrapper)
# ---------------------------------------------------------------------------

def resolve_bus_dir() -> Path:
    """
    Resolve the Pluribus bus directory path.

    Priority:
        1. PLURIBUS_BUS_DIR environment variable
        2. /pluribus/.pluribus/bus (standard location)
        3. ~/.pluribus/bus (user fallback)

    Returns:
        Path: Bus directory path (created if needed)
    """
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip()

    if bus_dir:
        bus_path = Path(bus_dir)
    elif Path("/pluribus/.pluribus/bus").parent.exists():
        bus_path = Path("/pluribus/.pluribus/bus")
    else:
        bus_path = Path.home() / ".pluribus" / "bus"

    bus_path.mkdir(parents=True, exist_ok=True)
    return bus_path


def emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    *,
    kind: str = "event",
    level: str = "info",
    actor: Optional[str] = None,
    trace_id: Optional[str] = None,
    bus_dir: Optional[Union[Path, str]] = None
) -> str:
    """
    Emit an event to the Pluribus bus.

    Args:
        topic: Event topic (e.g., "tools.operation.complete")
        data: Event payload dictionary
        kind: Event kind (event, request, response, metric, log)
        level: Log level (debug, info, warn, error)
        actor: Actor identifier (defaults to PLURIBUS_ACTOR env)
        trace_id: Correlation ID for distributed tracing
        bus_dir: Override bus directory (defaults to resolve_bus_dir())

    Returns:
        str: Generated event ID (UUID)

    Example:
        >>> event_id = emit_bus_event(
        ...     "remediation.step.complete",
        ...     {"step": 101, "status": "success"},
        ...     actor="subagent-beta"
        ... )
    """
    event_id = str(uuid.uuid4())

    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor or os.environ.get("PLURIBUS_ACTOR", "unknown"),
        "data": data,
    }

    # Add trace_id if provided or from environment
    if trace_id:
        event["trace_id"] = trace_id
    elif os.environ.get("PLURIBUS_TRACE_ID"):
        event["trace_id"] = os.environ.get("PLURIBUS_TRACE_ID")

    # Resolve bus directory
    if bus_dir:
        bus_path = Path(bus_dir) if isinstance(bus_dir, str) else bus_dir
    else:
        bus_path = resolve_bus_dir()

    events_file = bus_path / "events.ndjson"

    try:
        append_ndjson(events_file, event)
    except OSError as e:
        print(f"[helpers] Failed to emit bus event: {e}", file=sys.stderr)

    return event_id


# ---------------------------------------------------------------------------
# Actor/Environment Utilities
# ---------------------------------------------------------------------------

def get_actor() -> str:
    """
    Get the current actor identifier from environment.

    Returns:
        str: Actor name from PLURIBUS_ACTOR, USER, or "unknown"
    """
    return (
        os.environ.get("PLURIBUS_ACTOR") or
        os.environ.get("USER") or
        "unknown"
    )


def get_pluribus_root() -> Path:
    """
    Get the Pluribus root directory.

    Returns:
        Path: Root directory (default: /pluribus)
    """
    return Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))


# ---------------------------------------------------------------------------
# JSON Utilities
# ---------------------------------------------------------------------------

def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Parse JSON with fallback to default value on error.

    Args:
        text: JSON string to parse
        default: Value to return on parse error

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_read_json(path: Union[Path, str], default: Any = None) -> Any:
    """
    Read and parse JSON file with fallback on error.

    Args:
        path: Path to JSON file
        default: Value to return on read/parse error

    Returns:
        Parsed JSON or default value
    """
    p = Path(path) if isinstance(path, str) else path
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return default
    except (json.JSONDecodeError, OSError):
        return default


# ---------------------------------------------------------------------------
# UUID Utilities
# ---------------------------------------------------------------------------

def generate_event_id() -> str:
    """Generate a new UUID4 event identifier."""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a new UUID4 trace identifier for distributed tracing."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Timestamps
    "now_iso_utc",
    "now_epoch_ms",
    "parse_iso_utc",
    # Directory/File
    "ensure_dir",
    "ensure_parent_dir",
    # NDJSON
    "append_ndjson",
    "read_ndjson",
    # Bus
    "resolve_bus_dir",
    "emit_bus_event",
    # Actor/Environment
    "get_actor",
    "get_pluribus_root",
    # JSON
    "safe_json_loads",
    "safe_read_json",
    # UUID
    "generate_event_id",
    "generate_trace_id",
]
