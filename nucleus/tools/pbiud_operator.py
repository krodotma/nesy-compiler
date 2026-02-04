#!/usr/bin/env python3
"""
PBIUD Operator - Pluribus Iterate Unless Done
=============================================
Protocol: DKIN v28 (Conservation of Work)
Gate: SAGENT/SWAGENT

Summary:
  PBIUD is the "Definition of Done" operator. It ensures a task is:
  1. Implemented (Worktree clean or ready to commit)
  2. Verified (Optional PBTEST integration)
  3. Persisted (Commits to git via iso_git.mjs)
  4. Notified (Bus event emission)
  5. Chained (Suggests next co-semops)

Usage:
  python3 nucleus/tools/pbiud_operator.py [message] [options]

Options:
  --message, -m <msg>   Commit message (if changes exist).
  --verify              Run PBTEST --mode live before committing.
  --scope <scope>       Scope for verification/commit.
  --co-semops <op>      Explicitly trigger next operator (e.g. PBREALITY).
  --force               Skip interactive checks.

Bus Events:
  - operator.pbiud.start
  - operator.pbiud.complete
  - task.complete (if successful)
"""

import sys
import os
import argparse
import subprocess
import json
import time
from pathlib import Path

# Add repo root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from nucleus.tools.agent_bus import resolve_bus_paths, emit_event
except ImportError:
    # Fallback for standalone usage
    def resolve_bus_paths(d):
        return None
    def emit_event(*args, **kwargs): print(f"[BUS] {kwargs}")

ISO_GIT = "pluribus_next/tools/iso_git.mjs"
PLURIBUS_ROOT = "/pluribus"

def run_command(cmd, cwd=PLURIBUS_ROOT, capture=True):
    """Run a shell command."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=capture,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(e.stderr)
        return None

def check_status():
    """Check git status via iso_git."""
    cmd = f"node {ISO_GIT} status {PLURIBUS_ROOT}"
    return run_command(cmd)

def commit_changes(message, scope=None):
    """Commit changes via iso_git."""
    cmd = f"export PLURIBUS_ACTOR={os.environ.get('PLURIBUS_ACTOR', 'pbiud')} && node {ISO_GIT} commit {PLURIBUS_ROOT} \"{message}\""
    if scope:
        # If scope is provided, we might want to commit-paths, but PBIUD generally implies "done with task", so commit all is often appropriate.
        # For now, we'll stick to commit all for simplicity, or we could parse scope.
        pass
    
    print(f"Committing changes: {message}")
    return run_command(cmd, capture=False)

def verify_work(scope="."):
    """Run verification."""
    print(f"Verifying work in scope: {scope}")
    # Default to a lightweight check or prompt user
    # In full PBIUD, this would invoke PBTEST
    cmd = f"python3 nucleus/tools/pbtest_operator.py --scope {scope} --mode live --browser none --intent 'PBIUD Verification'"
    return run_command(cmd, capture=False)

def main():
    parser = argparse.ArgumentParser(description="PBIUD - Iterate Unless Done")
    parser.add_argument("message", nargs="?", help="Commit message")
    parser.add_argument("--verify", action="store_true", help="Run verification before commit")
    parser.add_argument("--scope", default=".", help="Scope for verification")
    parser.add_argument("--co-semops", help="Next operator to trigger")
    parser.add_argument("--force", action="store_true", help="Skip confirmations")
    
    args = parser.parse_args()
    
    bus_paths = resolve_bus_paths(None)
    actor = os.environ.get("PLURIBUS_ACTOR", "pbiud-operator")
    
    emit_event(bus_paths, topic="operator.pbiud.start", kind="log", level="info", actor=actor, data={"args": vars(args)}, trace_id=None, run_id=None, durable=True)

    print("PBIUD: Ensuring Task Completion...")

    # 1. Check Status
    status = check_status()
    if status and ("M " in status or "A " in status or "??" in status or "D " in status):
        print("Uncommitted changes detected.")
        if not args.message:
            print("Error: Uncommitted changes found but no commit message provided.")
            print("Usage: PBIUD \"commit message\"")
            sys.exit(1)
            
        # 2. Verify (Optional)
        if args.verify:
            if verify_work(args.scope) is None:
                print("Verification failed. Aborting PBIUD.")
                sys.exit(1)
        
        # 3. Commit
        if commit_changes(args.message) is None:
            print("Commit failed.")
            sys.exit(1)
    else:
        print("No uncommitted changes. Proceeding to notification.")

    # 4. Notify Bus
    emit_event(bus_paths, topic="task.complete", kind="artifact", level="info", actor=actor, data={
        "message": args.message or "Task marked done (no changes)",
        "status": "completed"
    }, trace_id=None, run_id=None, durable=True)
    
    # 5. Co-Semops
    next_ops = []
    if args.co_semops:
        next_ops.append(args.co_semops)
    else:
        # Heuristic for next steps
        next_ops.append("PBREALITY (Verify production parity)")
        next_ops.append("PBDEEP (Deep audit)")
    
    print(f"\nTask marked COMPLETE.")
    print("Suggested Co-Semops:")
    for op in next_ops:
        print(f"  - {op}")

    emit_event(bus_paths, topic="operator.pbiud.complete", kind="log", level="info", actor=actor, data={"next_ops": next_ops}, trace_id=None, run_id=None, durable=True)

if __name__ == "__main__":
    main()
