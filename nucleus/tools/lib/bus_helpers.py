#!/usr/bin/env python3
"""
Bus Helpers - Unified Event Emission for Pluribus Tools
========================================================

Provides centralized bus event emission with proper fallback handling,
thread safety, and DKIN v25 compliance.

This module eliminates code duplication across tools that emit bus events.

Usage:
    from nucleus.tools.lib.bus_helpers import emit_event, resolve_bus_path

    emit_event("topic.name", {"key": "value"}, level="info", actor="my-tool")
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
import fcntl
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    """Return current UTC time in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_bus_path() -> Path:
    """
    Resolve the Pluribus bus directory path with fallback logic.

    Priority:
    1. PLURIBUS_BUS_DIR environment variable
    2. /pluribus/.pluribus/bus (standard location)
    3. ~/.pluribus/bus (user fallback)

    Returns:
        Path to the bus directory (created if needed)
    """
    # Check environment variable first
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip()

    if bus_dir:
        bus_path = Path(bus_dir)
    else:
        # Standard location: /pluribus/.pluribus/bus
        standard_path = Path("/pluribus/.pluribus/bus")
        if standard_path.parent.exists():
            bus_path = standard_path
        else:
            # User fallback
            bus_path = Path.home() / ".pluribus" / "bus"

    # Ensure directory exists
    if not bus_path.exists():
        bus_path.mkdir(parents=True, exist_ok=True)

    return bus_path


def emit_event(
    topic: str,
    data: dict[str, Any],
    level: str = "info",
    actor: str | None = None,
    trace_id: str | None = None,
    kind: str = "event",
) -> str:
    """
    Emit an event to the Pluribus bus with atomic file locking.

    Args:
        topic: Event topic (e.g., "tools.lsp.operation.goto")
        data: Event payload dictionary
        level: Log level (debug, info, warn, error)
        actor: Actor identifier (defaults to PLURIBUS_ACTOR env or "unknown")
        trace_id: Optional correlation ID for distributed tracing
        kind: Event kind (event, request, response, metric)

    Returns:
        Event ID (UUID string)

    DKIN v25 Compliant - emits standardized bus events.
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

    bus_path = resolve_bus_path()
    events_file = bus_path / "events.ndjson"

    # Atomic write with file locking to prevent interleaving
    try:
        with events_file.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        # Best-effort - log to stderr if bus write fails
        print(f"[bus_helpers] Failed to emit event: {e}", file=sys.stderr)

    return event_id


def emit_metric(
    topic: str,
    value: float | int,
    unit: str = "",
    tags: dict[str, str] | None = None,
    actor: str | None = None,
) -> str:
    """
    Emit a metric event to the Pluribus bus.

    Args:
        topic: Metric topic (e.g., "telemetry.latency.api")
        value: Numeric metric value
        unit: Unit of measurement (ms, bytes, count, etc.)
        tags: Optional key-value tags for segmentation
        actor: Actor identifier

    Returns:
        Event ID
    """
    data = {
        "value": value,
        "unit": unit,
    }
    if tags:
        data["tags"] = tags

    return emit_event(topic, data, level="info", actor=actor, kind="metric")


def emit_request(
    topic: str,
    request_id: str,
    params: dict[str, Any],
    actor: str | None = None,
    trace_id: str | None = None,
) -> str:
    """
    Emit a request event for request/response tracking.

    Args:
        topic: Request topic
        request_id: Unique request identifier
        params: Request parameters
        actor: Actor identifier
        trace_id: Correlation ID

    Returns:
        Event ID
    """
    data = {
        "request_id": request_id,
        "params": params,
    }
    return emit_event(topic, data, actor=actor, trace_id=trace_id, kind="request")


def emit_response(
    topic: str,
    request_id: str,
    result: Any = None,
    error: dict[str, Any] | None = None,
    actor: str | None = None,
    trace_id: str | None = None,
) -> str:
    """
    Emit a response event for request/response tracking.

    Args:
        topic: Response topic
        request_id: Original request identifier
        result: Success result (None if error)
        error: Error details (None if success)
        actor: Actor identifier
        trace_id: Correlation ID

    Returns:
        Event ID
    """
    data: dict[str, Any] = {"request_id": request_id}
    if result is not None:
        data["result"] = result
    if error is not None:
        data["error"] = error

    level = "error" if error else "info"
    return emit_event(topic, data, level=level, actor=actor, trace_id=trace_id, kind="response")


# Convenience alias for backwards compatibility
emit_bus_event = emit_event
