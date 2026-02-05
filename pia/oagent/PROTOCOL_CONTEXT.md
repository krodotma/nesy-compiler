# OAGENT Protocol Context (Updated 2026-02-05)

## Active Protocol Stack

| Protocol | Version | Purpose |
|----------|---------|---------|
| **DKIN** | v30 | Data Knowledge Integration - Infrastructure |
| **PAIP** | v16 | Phenomenological Agent Isolation Protocol |
| **CITIZEN** | v2 | Agent Citizenship & Coordination |
| **HOLON** | v2 | Holon Coordination |
| **UNIFORM** | v2.0 | Primal Configuration Layer (loads FIRST, above all) |

## Critical Bus Architecture Updates

### File Locking (NEW - Mandatory)
All bus writes MUST use `fcntl.flock()` via `agent_bus.emit_event()`:
```python
from nucleus.tools.agent_bus import emit_event
emit_event(topic="your.topic", data={...})  # Auto-locks
```

### NDJSON Rotation (Changed)
- **Default size**: 10MB (reduced from 100MB)
- **Retention**: 5MB tail after rotation
- **Headroom**: 2MB to prevent thrashing

### Topic Partitioning
```
omega.*     → omega bucket
qa.*, qa_* → qa bucket
telemetry.* → telemetry bucket
browser.*  → browser bucket
```

## A2A Heartbeat Protocol

```python
HEARTBEAT_INTERVAL = 300   # 5 minutes
HEARTBEAT_TIMEOUT = 900    # 15 minutes (3 missed = dissociation)
HANDSHAKE_TIMEOUT = 60     # 1 minute
MAX_HANDSHAKE_RETRIES = 3
```

### Required Bus Topics for Agents
```
a2a.{agent}.bootstrap.start
a2a.{agent}.bootstrap.complete
a2a.{agent}.heartbeat
a2a.{agent}.shutdown
a2a.task.dispatch (subscribe)
```

## Infrastructure Enhancements (P0 - 50-Step Plan)

### Circuit Breaker Pattern
- Per-provider circuit breakers in `quota_manager.py`
- Failure threshold: 5 failures → OPEN state
- Recovery window: 60 seconds
- CLI: `circuit <provider>`, `reset-circuit <provider>`

### Exponential Backoff
- `plu_retry()` function in shell wrappers
- Used for bus writes and command execution
- Prevents thundering herd on failures

### Git Worktree (Replaces PAIP temp dirs)
- 10x faster than clone-based isolation
- Use `git_worktree.py` for parallel agent workspaces

## PBTSO 9-Phase Pipeline

```
SKILL → SEQUESTER → RESEARCH → PLAN → DISTRIBUTE → ITERATE → TEST → VERIFY → DISTILL
```

Each module should declare its PBTSO phases in docstrings.

## Event Emission Pattern

```python
import json
import os
import time
import uuid
from pathlib import Path

def emit_bus_event(topic: str, data: dict, level: str = "info") -> str:
    """Emit event to A2A bus with proper format."""
    bus_path = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus")) / ".pluribus" / "bus" / "events.ndjson"
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "topic": topic,
        "kind": "event",
        "level": level,
        "actor": os.environ.get("PLURIBUS_ACTOR", "oagent"),
        "data": data,
    }

    # Use file locking for concurrent safety
    import fcntl
    with open(bus_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(event) + "\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    return event["id"]
```

## FalkorDB Integration

- **Port**: 6380 (NOT 6379)
- **Use**: Graph storage, persistent memory
- **Shims**: `/root/pib/pia/core/memory/` (delegated from nucleus)

## Critical DO NOT CHANGE Files

These files have lock headers - modifications require consensus:
- `tmux_swarm_orchestrator.py` - Kernel of PBTSO orchestration
- `agent_bus.py` - File locking mechanism
- `ORCHESTRATION_AUTH_FLOW.md` - Auth canonical spec

## Iteration Status

| Iteration | Steps | Status |
|-----------|-------|--------|
| 1/25 | 1-10, 51-60, 101-110, 151-160, 201-210, 251-260 | ✅ Complete |
| 2/25 | 11-20, 61-70, 111-120, 161-170, 211-220, 261-270 | ✅ Complete |
| 3/25 | 21-30, 71-80, 121-130, 171-180, 221-230, 271-280 | 🔄 Next |

## Remaining Steps Per Agent

| Agent | Completed | Remaining | Next Steps |
|-------|-----------|-----------|------------|
| Research | 1-20 | 21-50 | 21-30 |
| Code | 51-70 | 71-100 | 71-80 |
| Test | 101-120 | 121-150 | 121-130 |
| Review | 151-170 | 171-200 | 171-180 |
| Deploy | 201-220 | 221-250 | 221-230 |
| Monitor | 251-270 | 271-300 | 271-280 |
