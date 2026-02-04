#!/usr/bin/env python3
"""
Clade Weaver (The Intelligence)
================================

A Ring-0 agent responsible for collapsing Clade superpositions into Main.
Implements the neurosymbolic merge protocol from CLADE_WEAVE.md.

Workflow:
1. Listen for `clade.lifecycle.propose` events on the bus
2. For each proposal:
   a. Analyze diffs between Clade and Main
   b. Fetch bus events (reasoning) associated with the Clade
   c. If conflicts: perform neurosymbolic synthesis
   d. If genotype touched: trigger constitutional review
   e. Run tests on merged candidate
   f. Commit result to main, archive clade as "fossil"

Bus Topics:
  - clade.lifecycle.propose (input)
  - clade.weaver.analysis (output: analysis results)
  - clade.weaver.merge.start (output)
  - clade.weaver.merge.complete (output)
  - clade.weaver.conflict (output: conflict detected)
  - clade.weaver.constitutional_review (output: genotype touched)
"""

import sys
import os
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

# Bootstrap imports
try:
    from agent_bus import emit_event, resolve_bus_paths
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from agent_bus import emit_event, resolve_bus_paths


def read_recent_events(bus_paths: dict, limit: int = 100, topic: str | None = None) -> list:
    """Read recent events from the bus (simple implementation)."""
    events_path = bus_paths.get("events_path")
    if not events_path or not Path(events_path).exists():
        return []

    events = []
    try:
        with open(events_path, 'r') as f:
            lines = f.readlines()[-limit:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if topic and topic not in event.get("topic", ""):
                    continue
                events.append(event)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return events

ISO_GIT = Path(__file__).resolve().parent / "iso_git.mjs"
GENOTYPE_LOCK = Path(__file__).resolve().parent.parent / "specs" / "genotype.lock"


@dataclass
class CladeAnalysis:
    """Result of analyzing a Clade proposal."""
    clade: str
    base_branch: str
    files_changed: List[str]
    commits: List[Dict[str, str]]
    has_conflicts: bool
    conflict_files: List[str]
    touches_genotype: bool
    genotype_files: List[str]
    reasoning_events: List[Dict[str, Any]]
    merge_strategy: str  # 'fast_forward', 'merge', 'neurosymbolic', 'constitutional'


@dataclass
class MergeResult:
    """Result of a merge operation."""
    success: bool
    strategy: str
    merged_sha: Optional[str]
    error: Optional[str]
    artifacts: List[Dict[str, Any]]


def run_iso(*args: str) -> subprocess.CompletedProcess:
    """Run iso_git.mjs with arguments."""
    cmd = ["node", str(ISO_GIT)] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent.parent))


def run_git(*args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run git command directly."""
    repo_dir = cwd or Path(__file__).resolve().parent.parent.parent
    return subprocess.run(["git"] + list(args), capture_output=True, text=True, cwd=str(repo_dir))


def load_genotype_lock() -> List[str]:
    """Load the list of protected genotype files."""
    if not GENOTYPE_LOCK.exists():
        return []
    with open(GENOTYPE_LOCK) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def analyze_clade(clade: str, base_branch: str = "main") -> CladeAnalysis:
    """Analyze a Clade branch for merge readiness."""
    bus_paths = resolve_bus_paths(None)
    genotype_files = load_genotype_lock()

    # Get files changed in the clade
    diff_result = run_git("diff", "--name-only", f"{base_branch}...{clade}")
    files_changed = [f.strip() for f in diff_result.stdout.splitlines() if f.strip()]

    # Get commit history
    log_result = run_git("log", f"{base_branch}..{clade}", "--format=%H|%s|%an|%ae", "--reverse")
    commits = []
    for line in log_result.stdout.splitlines():
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 4:
                commits.append({
                    "sha": parts[0],
                    "message": parts[1],
                    "author_name": parts[2],
                    "author_email": parts[3]
                })

    # Check for merge conflicts
    # First, try a dry-run merge
    merge_test = run_git("merge-tree", "--write-tree", base_branch, clade)
    has_conflicts = merge_test.returncode != 0 or "CONFLICT" in merge_test.stderr

    conflict_files = []
    if has_conflicts:
        # Parse conflict files from merge-tree output
        for line in merge_test.stderr.splitlines():
            if "CONFLICT" in line and ":" in line:
                conflict_files.append(line.split(":")[-1].strip())

    # Check if genotype files are touched
    touches_genotype = False
    genotype_touched = []
    for f in files_changed:
        for gf in genotype_files:
            if f == gf or f.startswith(gf.rstrip('/') + '/'):
                touches_genotype = True
                genotype_touched.append(f)

    # Fetch reasoning events from the bus
    # Look for events with trace_id matching the clade task_id
    reasoning_events = []
    task_id = clade.split('/')[-1] if '/' in clade else clade
    try:
        events = read_recent_events(bus_paths, limit=1000)
        reasoning_events = [e for e in events if task_id in str(e.get('trace_id', ''))]
    except Exception:
        pass

    # Determine merge strategy
    if touches_genotype:
        merge_strategy = "constitutional"
    elif has_conflicts:
        merge_strategy = "neurosymbolic"
    elif not commits:
        merge_strategy = "fast_forward"
    else:
        # Check if it can fast-forward
        ff_check = run_git("merge-base", "--is-ancestor", base_branch, clade)
        if ff_check.returncode == 0:
            merge_strategy = "fast_forward"
        else:
            merge_strategy = "merge"

    return CladeAnalysis(
        clade=clade,
        base_branch=base_branch,
        files_changed=files_changed,
        commits=commits,
        has_conflicts=has_conflicts,
        conflict_files=conflict_files,
        touches_genotype=touches_genotype,
        genotype_files=genotype_touched,
        reasoning_events=reasoning_events,
        merge_strategy=merge_strategy
    )


def fast_forward_merge(analysis: CladeAnalysis) -> MergeResult:
    """Perform a fast-forward merge."""
    result = run_git("checkout", analysis.base_branch)
    if result.returncode != 0:
        return MergeResult(False, "fast_forward", None, f"Checkout failed: {result.stderr}", [])

    result = run_git("merge", "--ff-only", analysis.clade)
    if result.returncode != 0:
        return MergeResult(False, "fast_forward", None, f"FF merge failed: {result.stderr}", [])

    # Get the new HEAD sha
    sha_result = run_git("rev-parse", "HEAD")
    return MergeResult(True, "fast_forward", sha_result.stdout.strip(), None, [])


def standard_merge(analysis: CladeAnalysis) -> MergeResult:
    """Perform a standard git merge."""
    result = run_git("checkout", analysis.base_branch)
    if result.returncode != 0:
        return MergeResult(False, "merge", None, f"Checkout failed: {result.stderr}", [])

    message = f"Merge clade {analysis.clade}\n\nCommits:\n" + "\n".join(
        f"  - {c['sha'][:7]}: {c['message']}" for c in analysis.commits
    )

    result = run_git("merge", "--no-ff", "-m", message, analysis.clade)
    if result.returncode != 0:
        # Abort the merge
        run_git("merge", "--abort")
        return MergeResult(False, "merge", None, f"Merge failed: {result.stderr}", [])

    sha_result = run_git("rev-parse", "HEAD")
    return MergeResult(True, "merge", sha_result.stdout.strip(), None, [])


def neurosymbolic_merge(analysis: CladeAnalysis) -> MergeResult:
    """
    Perform a neurosymbolic merge for conflicting changes.

    This is where the AI-assisted merge happens. The Weaver:
    1. Extracts the conflicting diffs
    2. Fetches the reasoning/intent from bus events
    3. Synthesizes a merged version that satisfies both intents
    4. Validates the synthesis
    """
    bus_paths = resolve_bus_paths(None)
    artifacts = []

    # Emit conflict detection event
    emit_event(
        bus_paths,
        topic="clade.weaver.conflict",
        kind="artifact",
        level="warning",
        actor="clade-weaver",
        data={
            "clade": analysis.clade,
            "conflict_files": analysis.conflict_files,
            "reasoning_event_count": len(analysis.reasoning_events)
        },
        durable=True
    )

    # For each conflicting file, we need to:
    # 1. Get base version
    # 2. Get clade version
    # 3. Get main version
    # 4. Extract intent from reasoning events
    # 5. Synthesize merged version

    # This is a placeholder for the actual neurosymbolic synthesis.
    # In practice, this would call an LLM with the diffs and reasoning.

    synthesis_request = {
        "clade": analysis.clade,
        "conflicts": [],
        "reasoning": [e.get("data", {}) for e in analysis.reasoning_events[:10]]
    }

    for conflict_file in analysis.conflict_files:
        # Get all three versions
        base_result = run_git("show", f"{analysis.base_branch}:{conflict_file}")
        clade_result = run_git("show", f"{analysis.clade}:{conflict_file}")

        synthesis_request["conflicts"].append({
            "file": conflict_file,
            "base": base_result.stdout if base_result.returncode == 0 else None,
            "clade": clade_result.stdout if clade_result.returncode == 0 else None,
        })

    # Emit synthesis request
    emit_event(
        bus_paths,
        topic="clade.weaver.synthesis.request",
        kind="request",
        level="info",
        actor="clade-weaver",
        data=synthesis_request,
        durable=True
    )

    artifacts.append({
        "type": "synthesis_request",
        "file_count": len(analysis.conflict_files),
        "reasoning_events": len(analysis.reasoning_events)
    })

    # In a full implementation, we would:
    # 1. Wait for synthesis response from an LLM agent
    # 2. Apply the synthesized changes
    # 3. Run tests
    # 4. If tests pass, commit and return success

    # For now, return a "pending" state indicating synthesis is needed
    return MergeResult(
        False,
        "neurosymbolic",
        None,
        "Neurosymbolic synthesis requested. Awaiting LLM agent response.",
        artifacts
    )


def constitutional_review(analysis: CladeAnalysis) -> MergeResult:
    """
    Trigger constitutional review for genotype modifications.

    Genotype files are protected and require special approval gates:
    - P: Policy alignment
    - E: Ethical review
    - L: Legal compliance
    - R: Risk assessment
    - Q: Quality assurance
    """
    bus_paths = resolve_bus_paths(None)

    # Emit constitutional review request
    emit_event(
        bus_paths,
        topic="clade.weaver.constitutional_review",
        kind="request",
        level="warning",
        actor="clade-weaver",
        data={
            "clade": analysis.clade,
            "genotype_files": analysis.genotype_files,
            "required_gates": ["P", "E", "L", "R", "Q"],
            "commits": analysis.commits,
            "reasoning_event_count": len(analysis.reasoning_events)
        },
        durable=True
    )

    return MergeResult(
        False,
        "constitutional",
        None,
        "Constitutional review required for genotype modifications. Awaiting gate approvals.",
        [{"type": "constitutional_review", "files": analysis.genotype_files}]
    )


def archive_clade(clade: str) -> bool:
    """Archive a merged clade as a 'fossil' tag."""
    timestamp = int(time.time())
    fossil_name = f"fossil/{clade.replace('/', '_')}_{timestamp}"

    result = run_git("tag", fossil_name, clade)
    if result.returncode != 0:
        return False

    # Optionally delete the branch
    # run_git("branch", "-d", clade)

    return True


def process_proposal(event: Dict[str, Any]) -> None:
    """Process a clade.lifecycle.propose event."""
    bus_paths = resolve_bus_paths(None)
    data = event.get("data", {})
    clade = data.get("clade", "")

    if not clade:
        print(f"[WEAVER] Invalid proposal: missing clade")
        return

    print(f"[WEAVER] Processing proposal for: {clade}")

    # Emit analysis start
    emit_event(
        bus_paths,
        topic="clade.weaver.analysis",
        kind="artifact",
        level="info",
        actor="clade-weaver",
        data={"clade": clade, "status": "started"},
        durable=True
    )

    # Analyze the clade
    try:
        analysis = analyze_clade(clade)
    except Exception as e:
        emit_event(
            bus_paths,
            topic="clade.weaver.analysis",
            kind="artifact",
            level="error",
            actor="clade-weaver",
            data={"clade": clade, "status": "failed", "error": str(e)},
            durable=True
        )
        return

    # Emit analysis results
    emit_event(
        bus_paths,
        topic="clade.weaver.analysis",
        kind="artifact",
        level="info",
        actor="clade-weaver",
        data={
            "clade": clade,
            "status": "complete",
            "analysis": {
                "files_changed": len(analysis.files_changed),
                "commits": len(analysis.commits),
                "has_conflicts": analysis.has_conflicts,
                "touches_genotype": analysis.touches_genotype,
                "merge_strategy": analysis.merge_strategy
            }
        },
        durable=True
    )

    print(f"[WEAVER] Analysis complete: strategy={analysis.merge_strategy}, conflicts={analysis.has_conflicts}")

    # Emit merge start
    emit_event(
        bus_paths,
        topic="clade.weaver.merge.start",
        kind="artifact",
        level="info",
        actor="clade-weaver",
        data={"clade": clade, "strategy": analysis.merge_strategy},
        durable=True
    )

    # Execute merge based on strategy
    if analysis.merge_strategy == "fast_forward":
        result = fast_forward_merge(analysis)
    elif analysis.merge_strategy == "merge":
        result = standard_merge(analysis)
    elif analysis.merge_strategy == "neurosymbolic":
        result = neurosymbolic_merge(analysis)
    elif analysis.merge_strategy == "constitutional":
        result = constitutional_review(analysis)
    else:
        result = MergeResult(False, "unknown", None, f"Unknown strategy: {analysis.merge_strategy}", [])

    # Emit merge result
    emit_event(
        bus_paths,
        topic="clade.weaver.merge.complete",
        kind="artifact",
        level="info" if result.success else "warning",
        actor="clade-weaver",
        data={
            "clade": clade,
            "success": result.success,
            "strategy": result.strategy,
            "merged_sha": result.merged_sha,
            "error": result.error,
            "artifacts": result.artifacts
        },
        durable=True
    )

    if result.success:
        print(f"[WEAVER] Merge successful: {result.merged_sha}")
        # Archive the clade
        if archive_clade(clade):
            print(f"[WEAVER] Clade archived as fossil")
    else:
        print(f"[WEAVER] Merge pending/failed: {result.error or 'awaiting action'}")


def daemon_loop() -> None:
    """Run the Weaver as a daemon, watching for proposals."""
    bus_paths = resolve_bus_paths(None)
    print(f"[WEAVER] Starting daemon (bus={bus_paths.get('active_dir', 'unknown')})")

    seen_ids = set()

    while True:
        try:
            # Tail recent events looking for proposals
            events = read_recent_events(bus_paths, limit=100, topic="clade.lifecycle.propose")

            for event in events:
                event_id = event.get("id")
                if event_id and event_id not in seen_ids:
                    seen_ids.add(event_id)
                    process_proposal(event)

            # Keep seen_ids from growing unbounded
            if len(seen_ids) > 10000:
                seen_ids = set(list(seen_ids)[-5000:])

            time.sleep(2)  # Poll interval

        except KeyboardInterrupt:
            print("\n[WEAVER] Shutting down")
            break
        except Exception as e:
            print(f"[WEAVER] Error: {e}")
            time.sleep(5)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Clade Weaver - Neurosymbolic Merge Agent")
    parser.add_argument("command", choices=["daemon", "analyze", "merge"], help="Command to run")
    parser.add_argument("--clade", help="Clade branch name (for analyze/merge)")
    parser.add_argument("--base", default="main", help="Base branch (default: main)")

    args = parser.parse_args()

    if args.command == "daemon":
        daemon_loop()
    elif args.command == "analyze":
        if not args.clade:
            print("Error: --clade required for analyze")
            sys.exit(1)
        analysis = analyze_clade(args.clade, args.base)
        print(json.dumps(asdict(analysis), indent=2, default=str))
    elif args.command == "merge":
        if not args.clade:
            print("Error: --clade required for merge")
            sys.exit(1)
        analysis = analyze_clade(args.clade, args.base)
        # Create a mock event
        process_proposal({"data": {"clade": args.clade}})


if __name__ == "__main__":
    main()
