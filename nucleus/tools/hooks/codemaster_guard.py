#!/usr/bin/env python3
"""
CODEMASTER GUARD - Pre-Push Hook for Branch Protection
=======================================================

E Pluribus Unum - "From Many, One"

This hook prevents direct pushes to critical branches (main, staging, dev).
All merges to these branches must go through the Codemaster Agent via PBCMASTER.

Installation:
    cp nucleus/tools/hooks/codemaster_guard.py .git/hooks/pre-push
    chmod +x .git/hooks/pre-push

Or via Claude Code hooks:
    # In .claude-code/hooks.json
    {
      "pre_push": ["python3 nucleus/tools/hooks/codemaster_guard.py"]
    }

Reference: nucleus/specs/codemaster_protocol_v2.md
DKIN Version: v26
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Critical branches that require Codemaster
CRITICAL_BRANCHES = {"main", "staging", "dev"}

# Environment variable to bypass guard (only for Codemaster process)
BYPASS_VAR = "CODEMASTER_BYPASS"

# Bus directory
DEFAULT_BUS_DIR = "/pluribus/.pluribus/bus"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_violation(
    bus_dir: Path,
    branch: str,
    actor: str,
    remote: str,
) -> str:
    """Emit violation event to bus."""
    events_path = bus_dir / "events.ndjson"
    bus_dir.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())

    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": now_iso(),
        "topic": "codemaster.violation.direct_push",
        "kind": "alert",
        "level": "warn",
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": {
            "branch": branch,
            "remote": remote,
            "blocked": True,
            "message": f"Direct push to protected branch '{branch}' blocked. Use PBCMASTER.",
        },
    }

    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"

    try:
        with events_path.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass

    return event_id


def is_codemaster_process() -> bool:
    """Check if current process is Codemaster."""
    return os.environ.get(BYPASS_VAR) == "1"


def get_actor() -> str:
    """Get current actor name."""
    return os.environ.get("PLURIBUS_ACTOR", os.environ.get("USER", "unknown"))


def parse_push_refs(stdin_lines: list[str]) -> list[tuple[str, str, str, str]]:
    """
    Parse pre-push hook stdin.

    Format: <local ref> <local sha1> <remote ref> <remote sha1>
    Returns list of (local_ref, local_sha, remote_ref, remote_sha)
    """
    refs = []
    for line in stdin_lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            refs.append((parts[0], parts[1], parts[2], parts[3]))
    return refs


def get_branch_from_ref(ref: str) -> Optional[str]:
    """Extract branch name from ref."""
    prefixes = ["refs/heads/", "refs/remotes/origin/"]
    for prefix in prefixes:
        if ref.startswith(prefix):
            return ref[len(prefix):]
    return ref


def main() -> int:
    """Main entry point for pre-push hook."""
    # Check bypass
    if is_codemaster_process():
        return 0

    # Read remote and URL from args
    remote = sys.argv[1] if len(sys.argv) > 1 else "origin"
    # url = sys.argv[2] if len(sys.argv) > 2 else ""

    # Read refs from stdin
    stdin_lines = sys.stdin.read().strip().split("\n") if not sys.stdin.isatty() else []
    refs = parse_push_refs(stdin_lines)

    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", DEFAULT_BUS_DIR))
    actor = get_actor()
    blocked = False

    for local_ref, local_sha, remote_ref, remote_sha in refs:
        branch = get_branch_from_ref(remote_ref)

        if branch in CRITICAL_BRANCHES:
            # Block push to critical branch
            blocked = True

            # Emit violation event
            emit_violation(bus_dir, branch, actor, remote)

            # Print error message
            print(f"\n{'=' * 70}", file=sys.stderr)
            print(f"CODEMASTER GUARD: Direct push to '{branch}' blocked!", file=sys.stderr)
            print(f"{'=' * 70}", file=sys.stderr)
            print(f"\nE Pluribus Unum - From Many, One", file=sys.stderr)
            print(f"\nAll merges to critical branches must go through Codemaster.", file=sys.stderr)
            print(f"\nTo merge your changes:", file=sys.stderr)
            print(f"  1. Push to a feature branch first", file=sys.stderr)
            print(f"  2. Request merge via PBCMASTER:", file=sys.stderr)
            print(f"\n     PBCMASTER merge --source <your-branch> --target {branch}", file=sys.stderr)
            print(f"\nReference: nucleus/specs/codemaster_protocol_v2.md", file=sys.stderr)
            print(f"{'=' * 70}\n", file=sys.stderr)

    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
