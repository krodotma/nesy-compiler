# Pluribus Tools Library
# Shared utilities for all nucleus tools

"""
nucleus.tools.lib - Shared Utilities for Pluribus Tools
========================================================

This package provides consolidated utility functions to eliminate
code duplication across the nucleus toolchain.

Modules:
    - bus_helpers: Bus event emission with atomic file locking
    - helpers: General-purpose utilities (timestamps, NDJSON, etc.)

Usage:
    from nucleus.tools.lib.helpers import now_iso_utc, ensure_dir, emit_bus_event
    from nucleus.tools.lib.bus_helpers import emit_event, emit_metric

DKIN v28 Compliant.
"""

# Re-export commonly used functions for convenience
from nucleus.tools.lib.helpers import (
    now_iso_utc,
    now_epoch_ms,
    ensure_dir,
    ensure_parent_dir,
    append_ndjson,
    read_ndjson,
    emit_bus_event,
    resolve_bus_dir,
    get_actor,
    get_pluribus_root,
    safe_json_loads,
    safe_read_json,
    generate_event_id,
    generate_trace_id,
)

from nucleus.tools.lib.bus_helpers import (
    emit_event,
    emit_metric,
    emit_request,
    emit_response,
    resolve_bus_path,
)

__all__ = [
    # From helpers
    "now_iso_utc",
    "now_epoch_ms",
    "ensure_dir",
    "ensure_parent_dir",
    "append_ndjson",
    "read_ndjson",
    "emit_bus_event",
    "resolve_bus_dir",
    "get_actor",
    "get_pluribus_root",
    "safe_json_loads",
    "safe_read_json",
    "generate_event_id",
    "generate_trace_id",
    # From bus_helpers
    "emit_event",
    "emit_metric",
    "emit_request",
    "emit_response",
    "resolve_bus_path",
]
