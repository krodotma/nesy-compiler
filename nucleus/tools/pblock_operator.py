#!/usr/bin/env python3
from __future__ import annotations

"""
PBLOCK — Milestone Freeze Operator (Protocol v16)

PBLOCK is a multiagent coordination primitive that enforces a milestone checkpoint:
  - When PBLOCK=true:
    - NO new features (reject feat: commits)
    - NO implementation extensions
    - DO iterate on incomplete/unimplemented tasks
    - DO run e2e tests with full coverage
  - Exit criteria:
    - All tests pass
    - All work pushed to local remotes + GitHub
    - PBLOCK=false restored to collective working memory

State is persisted to a JSON file and broadcast via bus events.
All agents read this state to enforce guards.
"""

import argparse
import getpass
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

PBLOCK_STATE_FILE = Path("/var/lib/pluribus/.pluribus/pblock_state.json")
PBLOCK_PROTOCOL_VERSION = 16


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    evt = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)


def load_state(state_file: Path) -> dict[str, Any]:
    """Load PBLOCK state from file, or return default inactive state."""
    if state_file.exists():
        try:
            with state_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "active": False,
        "entered_iso": None,
        "entered_by": None,
        "milestone": None,
        "reason": None,
        "incomplete_tasks": [],
        "test_coverage": None,
        "exit_criteria": {
            "all_tests_pass": False,
            "pushed_to_remotes": False,
            "pushed_to_github": False,
        },
        "protocol_version": PBLOCK_PROTOCOL_VERSION,
    }


def save_state(state_file: Path, state: dict[str, Any]) -> None:
    """Persist PBLOCK state to file."""
    ensure_dir(state_file.parent)
    state["last_updated_iso"] = now_iso_utc()
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def enter_pblock(
    state_file: Path,
    bus_dir: Path,
    actor: str,
    milestone: str | None,
    reason: str | None,
) -> dict[str, Any]:
    """Enter PBLOCK freeze state."""
    state = load_state(state_file)
    if state["active"]:
        return {"error": "PBLOCK already active", "state": state}

    state["active"] = True
    state["entered_iso"] = now_iso_utc()
    state["entered_by"] = actor
    state["milestone"] = milestone or f"milestone-{int(time.time())}"
    state["reason"] = reason or "milestone checkpoint"
    state["exit_criteria"] = {
        "all_tests_pass": False,
        "pushed_to_remotes": False,
        "pushed_to_github": False,
    }
    state["protocol_version"] = PBLOCK_PROTOCOL_VERSION

    save_state(state_file, state)

    # Emit bus events
    payload = {
        "req_id": str(uuid.uuid4()),
        "action": "enter",
        "milestone": state["milestone"],
        "reason": state["reason"],
        "actor": actor,
        "iso": state["entered_iso"],
        "protocol_version": PBLOCK_PROTOCOL_VERSION,
    }
    emit_bus(bus_dir, topic="operator.pblock.enter", kind="request", level="warn", actor=actor, data=payload)
    emit_bus(bus_dir, topic="operator.pblock.state", kind="metric", level="info", actor=actor, data={"active": True, **payload})
    emit_bus(bus_dir, topic="infer_sync.request", kind="request", level="info", actor=actor, data={
        "req_id": payload["req_id"],
        "intent": "pblock_enter",
        "subproject": "ops",
        "message": f"PBLOCK entered: {state['milestone']} - {state['reason']}",
    })

    return {"success": True, "state": state}


def exit_pblock(
    state_file: Path,
    bus_dir: Path,
    actor: str,
    force: bool = False,
) -> dict[str, Any]:
    """Exit PBLOCK freeze state."""
    state = load_state(state_file)
    if not state["active"]:
        return {"error": "PBLOCK not active", "state": state}

    if not force:
        # Check exit criteria
        criteria = state.get("exit_criteria", {})
        if not criteria.get("all_tests_pass"):
            return {"error": "Cannot exit PBLOCK: tests not passing. Use --force-exit to override.", "state": state}

    exit_iso = now_iso_utc()
    duration_s = None
    if state.get("entered_iso"):
        try:
            entered_ts = time.mktime(time.strptime(state["entered_iso"], "%Y-%m-%dT%H:%M:%SZ"))
            exit_ts = time.mktime(time.strptime(exit_iso, "%Y-%m-%dT%H:%M:%SZ"))
            duration_s = int(exit_ts - entered_ts)
        except (ValueError, TypeError):
            pass

    milestone = state.get("milestone")
    state["active"] = False
    state["exited_iso"] = exit_iso
    state["exited_by"] = actor
    state["duration_seconds"] = duration_s

    save_state(state_file, state)

    # Emit bus events
    payload = {
        "req_id": str(uuid.uuid4()),
        "action": "exit",
        "milestone": milestone,
        "actor": actor,
        "iso": exit_iso,
        "duration_seconds": duration_s,
        "forced": force,
        "protocol_version": PBLOCK_PROTOCOL_VERSION,
    }
    emit_bus(bus_dir, topic="operator.pblock.exit", kind="request", level="info", actor=actor, data=payload)
    emit_bus(bus_dir, topic="operator.pblock.state", kind="metric", level="info", actor=actor, data={"active": False, **payload})
    emit_bus(bus_dir, topic="infer_sync.request", kind="request", level="info", actor=actor, data={
        "req_id": payload["req_id"],
        "intent": "pblock_exit",
        "subproject": "ops",
        "message": f"PBLOCK exited: {milestone} (duration: {duration_s}s, forced: {force})",
    })

    return {"success": True, "state": state}


def update_criteria(
    state_file: Path,
    bus_dir: Path,
    actor: str,
    tests_pass: bool | None = None,
    pushed_remotes: bool | None = None,
    pushed_github: bool | None = None,
    test_coverage: float | None = None,
) -> dict[str, Any]:
    """Update exit criteria for PBLOCK."""
    state = load_state(state_file)
    if not state["active"]:
        return {"error": "PBLOCK not active", "state": state}

    criteria = state.get("exit_criteria", {})
    if tests_pass is not None:
        criteria["all_tests_pass"] = tests_pass
    if pushed_remotes is not None:
        criteria["pushed_to_remotes"] = pushed_remotes
    if pushed_github is not None:
        criteria["pushed_to_github"] = pushed_github
    if test_coverage is not None:
        state["test_coverage"] = test_coverage
    state["exit_criteria"] = criteria

    save_state(state_file, state)

    # Emit state update
    emit_bus(bus_dir, topic="operator.pblock.state", kind="metric", level="info", actor=actor, data={
        "active": True,
        "milestone": state.get("milestone"),
        "exit_criteria": criteria,
        "test_coverage": state.get("test_coverage"),
        "protocol_version": PBLOCK_PROTOCOL_VERSION,
    })

    return {"success": True, "state": state}


def check_guard(commit_prefix: str) -> dict[str, Any]:
    """Check if a commit prefix is allowed under PBLOCK."""
    state = load_state(PBLOCK_STATE_FILE)
    if not state["active"]:
        return {"allowed": True, "pblock_active": False}

    prefix = commit_prefix.lower().strip()
    allowed_prefixes = ["fix", "test", "refactor", "docs", "chore", "ci", "perf", "style"]
    blocked_prefixes = ["feat", "feature", "add", "new", "extend", "implement"]

    if any(prefix.startswith(b) for b in blocked_prefixes):
        return {
            "allowed": False,
            "pblock_active": True,
            "reason": f"PBLOCK active: '{commit_prefix}' blocked. Only fix/test/refactor/docs/chore allowed.",
            "milestone": state.get("milestone"),
        }

    if any(prefix.startswith(a) for a in allowed_prefixes):
        return {"allowed": True, "pblock_active": True}

    # Unknown prefix - allow but warn
    return {"allowed": True, "pblock_active": True, "warning": f"Unknown prefix '{commit_prefix}' - verify manually."}


def format_status(state: dict[str, Any]) -> str:
    """Format PBLOCK state for display."""
    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════════════╗")
    if state["active"]:
        lines.append("║  🔒 PBLOCK ACTIVE — MILESTONE FREEZE                                 ║")
    else:
        lines.append("║  🔓 PBLOCK INACTIVE — NORMAL DEVELOPMENT                            ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  Protocol: DKIN v{PBLOCK_PROTOCOL_VERSION}                                                      ║")
    lines.append(f"║  Active: {str(state.get('active', False)):5s}                                                    ║")

    if state["active"]:
        lines.append(f"║  Milestone: {str(state.get('milestone', '—'))[:50]:50s}     ║")
        lines.append(f"║  Entered: {str(state.get('entered_iso', '—'))[:25]:25s}                         ║")
        lines.append(f"║  By: {str(state.get('entered_by', '—'))[:20]:20s}                                    ║")
        lines.append(f"║  Reason: {str(state.get('reason', '—'))[:50]:50s}      ║")
        lines.append("╠══════════════════════════════════════════════════════════════════════╣")
        lines.append("║  GUARDS:                                                             ║")
        lines.append("║    ❌ feat:, feature:, add:, extend: → BLOCKED                       ║")
        lines.append("║    ✅ fix:, test:, refactor:, docs: → ALLOWED                        ║")
        lines.append("╠══════════════════════════════════════════════════════════════════════╣")
        lines.append("║  EXIT CRITERIA:                                                      ║")
        criteria = state.get("exit_criteria", {})
        tests_ok = "✅" if criteria.get("all_tests_pass") else "❌"
        remotes_ok = "✅" if criteria.get("pushed_to_remotes") else "❌"
        github_ok = "✅" if criteria.get("pushed_to_github") else "❌"
        lines.append(f"║    {tests_ok} All tests pass                                             ║")
        lines.append(f"║    {remotes_ok} Pushed to local remotes                                   ║")
        lines.append(f"║    {github_ok} Pushed to GitHub                                           ║")
        coverage = state.get("test_coverage")
        if coverage is not None:
            lines.append(f"║    📊 Coverage: {coverage:.1f}%                                            ║")
    else:
        exited = state.get("exited_iso")
        if exited:
            lines.append(f"║  Last exited: {str(exited)[:25]:25s}                       ║")
            duration = state.get("duration_seconds")
            if duration:
                lines.append(f"║  Duration: {duration}s                                                  ║")

    lines.append("╚══════════════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pblock_operator.py",
        description="PBLOCK — Milestone Freeze Operator (Protocol v16)",
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    p.add_argument("--actor", default=None, help="Actor identity")
    p.add_argument("--state-file", default=None, help="State file path")

    action = p.add_mutually_exclusive_group()
    action.add_argument("--enter", action="store_true", help="Enter PBLOCK freeze state")
    action.add_argument("--exit", action="store_true", help="Exit PBLOCK freeze state")
    action.add_argument("--status", action="store_true", help="Show current PBLOCK state")
    action.add_argument("--check-guard", metavar="PREFIX", help="Check if commit prefix is allowed")
    action.add_argument("--update-criteria", action="store_true", help="Update exit criteria")

    p.add_argument("--milestone", default=None, help="Milestone name/tag")
    p.add_argument("--reason", default=None, help="Reason for entering PBLOCK")
    p.add_argument("--force-exit", action="store_true", help="Force exit without verification")
    p.add_argument("--tests-pass", action="store_true", help="Mark tests as passing")
    p.add_argument("--pushed-remotes", action="store_true", help="Mark pushed to remotes")
    p.add_argument("--pushed-github", action="store_true", help="Mark pushed to GitHub")
    p.add_argument("--coverage", type=float, help="Test coverage percentage")
    p.add_argument("--json", action="store_true", help="JSON output")

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    actor = (args.actor or default_actor()).strip() or "pblock"
    bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else Path(
        os.environ.get("PLURIBUS_BUS_DIR") or "/var/lib/pluribus/.pluribus/bus"
    ).expanduser().resolve()
    state_file = Path(args.state_file).expanduser().resolve() if args.state_file else PBLOCK_STATE_FILE

    ensure_dir(bus_dir)
    (bus_dir / "events.ndjson").touch(exist_ok=True)

    result: dict[str, Any] = {}

    if args.enter:
        result = enter_pblock(state_file, bus_dir, actor, args.milestone, args.reason)
    elif args.exit:
        result = exit_pblock(state_file, bus_dir, actor, force=args.force_exit)
    elif args.check_guard:
        result = check_guard(args.check_guard)
    elif args.update_criteria:
        result = update_criteria(
            state_file, bus_dir, actor,
            tests_pass=args.tests_pass or None,
            pushed_remotes=args.pushed_remotes or None,
            pushed_github=args.pushed_github or None,
            test_coverage=args.coverage,
        )
    else:
        # Default: status
        state = load_state(state_file)
        result = {"state": state}

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        if "error" in result:
            sys.stderr.write(f"ERROR: {result['error']}\n")
            return 1
        if "state" in result:
            sys.stdout.write(format_status(result["state"]) + "\n")
        elif "allowed" in result:
            if result["allowed"]:
                sys.stdout.write(f"✅ ALLOWED: {args.check_guard}\n")
            else:
                sys.stdout.write(f"❌ BLOCKED: {result.get('reason', 'PBLOCK active')}\n")
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
