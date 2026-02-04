#!/usr/bin/env python3
"""
Clade Manager (The Interface)
=============================

Manages the lifecycle of Evolutionary Clades (Git Branches).
Wraps `iso_git.mjs` to enforce the Clade-Weave Protocol.

This module is a lightweight wrapper that delegates to the full
Clade Manager Protocol (CMP) implementation in clade_registry.py.

Commands:
  start <task_id>   - Create a new clade branch (delegates to CMP speciate)
  propose           - Signal intent to merge (emits bus event)
  list              - Show active clades
  status            - Show clade status (delegates to CMP)
  evaluate          - Evaluate clade fitness (delegates to CMP)

For full CMP functionality, use clade_cli.py directly:
  python3 clade_cli.py speciate <clade_id> --parent main --pressure "feature"
  python3 clade_cli.py evaluate --all
  python3 clade_cli.py recommend-merge
"""

import sys
import argparse
import subprocess
import json
import os
from pathlib import Path

# Bootstrap imports
try:
    from agent_bus import emit_bus_event as emit_event, resolve_bus_paths
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from agent_bus import emit_bus_event as emit_event, resolve_bus_paths

# Import CMP registry (optional - graceful fallback)
try:
    from clade_registry import CladeRegistry
    HAS_CMP = True
except ImportError:
    HAS_CMP = False

ISO_GIT = Path(__file__).resolve().parent / "iso_git.mjs"

def run_iso(args):
    cmd = ["node", str(ISO_GIT)] + args
    return subprocess.run(cmd, capture_output=True, text=True)

def cmd_start(args):
    """Create a new clade branch and register with CMP."""
    task_id = args.task_id
    agent = os.environ.get("PLURIBUS_ACTOR", "unknown")
    branch_name = f"clade/{agent}/{task_id}"
    clade_id = f"{agent}/{task_id}" if "/" not in task_id else task_id

    print(f"Starting Clade: {branch_name}")

    # 1. Create Branch: node iso_git.mjs branch . <name>
    res = run_iso(["branch", ".", branch_name])
    if res.returncode != 0:
        print(f"Error creating branch: {res.stderr}")
        return

    # 2. Checkout: node iso_git.mjs checkout . <name>
    res = run_iso(["checkout", ".", branch_name])
    if res.returncode != 0:
        print(f"Error checking out: {res.stderr}")
        return

    # 3. Register with CMP if available
    if HAS_CMP:
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            registry = CladeRegistry(repo_root)
            registry.load()

            # Only register if not already tracked
            if not registry.get_clade(clade_id):
                pressure = args.pressure if hasattr(args, 'pressure') and args.pressure else f"task-{task_id}"
                registry.speciate(
                    clade_id=clade_id,
                    parent="main",
                    pressure=pressure,
                    mutation_rate=0.3
                )
                registry.save()
                print(f"Registered with CMP: {clade_id}")
        except Exception as e:
            print(f"Warning: CMP registration skipped: {e}")

    # 4. Emit Event (legacy bus topic for backwards compatibility)
    emit_event(
        resolve_bus_paths(None),
        topic="clade.lifecycle.start",
        kind="request",
        level="info",
        actor=agent,
        data={"clade": branch_name, "task_id": task_id},
        trace_id=task_id,
        durable=True
    )
    print("Clade initialized.")

def cmd_propose(args):
    """Signal intent to merge the current clade."""
    # Get current branch: node iso_git.mjs branch .
    res = run_iso(["branch", "."])
    # Output format is list of branches with * current
    current_branch = None
    for line in res.stdout.splitlines():
        if line.strip().startswith("* "):
            current_branch = line.strip()[2:]
            break

    if not current_branch or not current_branch.startswith("clade/"):
        print(f"Error: Not in a Clade branch ({current_branch}).")
        return

    print(f"Proposing Clade: {current_branch}")

    # Update CMP status to converging if available
    if HAS_CMP:
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            registry = CladeRegistry(repo_root)
            registry.load()

            # Extract clade_id from branch name (clade/agent/task -> agent/task)
            clade_id = "/".join(current_branch.split("/")[1:])
            clade = registry.get_clade(clade_id)
            if clade and clade.status == "active":
                registry.update_status(clade_id, "converging")
                registry.save()
                print(f"CMP status updated: {clade_id} -> converging")
        except Exception as e:
            print(f"Warning: CMP update skipped: {e}")

    # Emit Proposal (legacy bus topic)
    emit_event(
        resolve_bus_paths(None),
        topic="clade.lifecycle.propose",
        kind="request",
        level="info",
        actor=os.environ.get("PLURIBUS_ACTOR", "unknown"),
        data={"clade": current_branch},
        durable=True
    )
    print("Proposal emitted. Awaiting Weaver.")

def cmd_list(args):
    """List all clades."""
    if HAS_CMP:
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            registry = CladeRegistry(repo_root)
            registry.load()

            clades = registry.list_clades()
            if not clades:
                print("No clades registered")
                return

            print(f"{'CLADE':<25} {'STATUS':<12} {'FITNESS':<10} {'PARENT'}")
            print("-" * 60)
            for c in clades:
                fitness = f"{c.fitness.score:.3f}" if c.fitness else "-"
                print(f"{c.clade_id:<25} {c.status:<12} {fitness:<10} {c.parent}")
            return
        except Exception as e:
            print(f"CMP error: {e}")

    # Fallback: list git branches
    res = run_iso(["branch", "."])
    clade_branches = [
        line.strip().lstrip("* ")
        for line in res.stdout.splitlines()
        if "clade/" in line
    ]

    if clade_branches:
        print("Clade branches:")
        for b in clade_branches:
            print(f"  {b}")
    else:
        print("No clade branches found")

def cmd_status(args):
    """Show detailed clade status (delegates to CMP)."""
    if not HAS_CMP:
        print("CMP not available. Use clade_cli.py for full functionality.")
        return

    try:
        repo_root = Path(__file__).resolve().parent.parent.parent
        registry = CladeRegistry(repo_root)
        registry.load()

        if hasattr(args, 'clade_id') and args.clade_id:
            clade = registry.get_clade(args.clade_id)
            if clade:
                print(f"Clade: {clade.clade_id}")
                print(f"  Status: {clade.status}")
                print(f"  Parent: {clade.parent}")
                print(f"  Generation: {clade.generation}")
                print(f"  Pressure: {clade.selection_pressure}")
                if clade.fitness:
                    print(f"  Fitness: {clade.fitness.score:.3f}")
            else:
                print(f"Clade not found: {args.clade_id}")
        else:
            cmd_list(args)
    except Exception as e:
        print(f"Error: {e}")

def cmd_evaluate(args):
    """Evaluate clade fitness (delegates to CMP)."""
    if not HAS_CMP:
        print("CMP not available. Use clade_cli.py for full functionality.")
        return

    try:
        repo_root = Path(__file__).resolve().parent.parent.parent
        registry = CladeRegistry(repo_root)
        registry.load()

        clade_id = args.clade_id
        clade = registry.get_clade(clade_id)
        if not clade:
            print(f"Clade not found: {clade_id}")
            return

        fitness = registry.evaluate_fitness(clade_id)
        registry.save()

        print(f"Fitness for {clade_id}: {fitness.score:.3f}")
        print(f"  Task completion: {fitness.task_completion:.0%}")
        print(f"  Test coverage:   {fitness.test_coverage:.0%}")
        print(f"  Bug rate:        {fitness.bug_rate:.0%}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Clade Manager - Evolutionary Branch Management",
        epilog="For full CMP functionality: python3 clade_cli.py --help"
    )
    subparsers = parser.add_subparsers()

    # start (legacy)
    p_start = subparsers.add_parser("start", help="Create a new clade branch")
    p_start.add_argument("task_id", help="Task identifier")
    p_start.add_argument("--pressure", help="Selection pressure (optional)")
    p_start.set_defaults(func=cmd_start)

    # propose (legacy)
    p_propose = subparsers.add_parser("propose", help="Signal intent to merge")
    p_propose.set_defaults(func=cmd_propose)

    # list (enhanced with CMP)
    p_list = subparsers.add_parser("list", help="List all clades")
    p_list.set_defaults(func=cmd_list)

    # status (CMP)
    p_status = subparsers.add_parser("status", help="Show clade status")
    p_status.add_argument("clade_id", nargs="?", help="Specific clade ID")
    p_status.set_defaults(func=cmd_status)

    # evaluate (CMP)
    p_eval = subparsers.add_parser("evaluate", help="Evaluate clade fitness")
    p_eval.add_argument("clade_id", help="Clade to evaluate")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
