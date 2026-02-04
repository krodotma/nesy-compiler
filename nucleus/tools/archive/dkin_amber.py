#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import json
import time
from datetime import datetime, timezone

# DKIN Amber Preservation Operator
# Manages proactive "Amber" snapshots (dangling commits) for erasure protection.

AMBER_REF_PREFIX = "refs/amber/"
ISO_AMBER_SCRIPT = os.path.join(os.path.dirname(__file__), "iso_amber.mjs")

def run_command(cmd, cwd=None):
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=30
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}", file=sys.stderr)
        raise

def trigger_snapshot(repo_dir, message, agent_id):
    """
    Triggers the iso_amber.mjs script to create a snapshot commit.
    Then creates a ref pointing to it: refs/amber/<agent_id>/<timestamp>
    """
    if not os.path.exists(ISO_AMBER_SCRIPT):
        raise FileNotFoundError(f"Missing iso_amber.mjs at {ISO_AMBER_SCRIPT}")

    print(f"Creating Amber snapshot for {agent_id}...", file=sys.stderr) 
    
    # 1. Create the commit object (dangling)
    # We execute in repo_dir, so we tell iso_amber to use current dir (.)
    commit_sha = run_command(["node", ISO_AMBER_SCRIPT, ".", message], cwd=repo_dir)
    
    if not commit_sha or len(commit_sha) != 40:
        raise ValueError(f"Invalid SHA returned from iso_amber: {commit_sha}")

    # 2. Create the ref
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ref_name = f"{AMBER_REF_PREFIX}{agent_id}/{timestamp}"
    
    # git update-ref refs/amber/... <sha>
    # Note: Using direct git CLI here because iso-git wrapper might not expose update-ref easily via CLI yet.
    # But wait, we are not supposed to use 'git'. We should use 'iso_git.mjs' if possible.
    # However, for this low-level plumbing, we might need to assume 'git' is available in the shell 
    # OR extend iso_git.mjs.
    # Given the constraint "Direct git CLI usage is forbidden" (IsoGit manages phenotype),
    # I should strictly rely on iso_git.mjs capabilities or file writes.
    
    # IsoGit writes refs as files in .git/refs/...
    # Let's write the ref file directly to adhere to "No git CLI".
    git_dir = os.path.join(repo_dir, ".git")
    ref_path = os.path.join(git_dir, ref_name)
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    
    with open(ref_path, "w") as f:
        f.write(commit_sha + "\n")
        
    print(json.dumps({
        "status": "success",
        "sha": commit_sha,
        "ref": ref_name,
        "timestamp": timestamp
    }))

def list_snapshots(repo_dir, agent_id=None):
    """
    Lists all amber snapshots found in .git/refs/amber
    """
    amber_root = os.path.join(repo_dir, ".git", "refs", "amber")
    snapshots = []
    
    if not os.path.exists(amber_root):
        print("[]")
        return

    for root, dirs, files in os.walk(amber_root):
        for file in files:
            full_path = os.path.join(root, file)
            # path relative to amber_root
            rel_path = os.path.relpath(full_path, amber_root)
            # rel_path is likely <agent_id>/<timestamp>
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                found_agent = parts[0]
                ts = parts[-1]
                
                if agent_id and found_agent != agent_id:
                    continue
                
                with open(full_path, 'r') as f:
                    sha = f.read().strip()
                
                snapshots.append({
                    "agent": found_agent,
                    "timestamp": ts,
                    "sha": sha,
                    "ref": f"refs/amber/{rel_path}"
                })
    
    # Sort by timestamp desc
    snapshots.sort(key=lambda x: x['timestamp'], reverse=True)
    print(json.dumps(snapshots, indent=2))

def main():
    parser = argparse.ArgumentParser(description="DKIN Amber Preservation Tool")
    subparsers = parser.add_subparsers(dest="command")

    trigger_parser = subparsers.add_parser("trigger")
    trigger_parser.add_argument("--repo", default=".")
    trigger_parser.add_argument("--message", default="Amber Auto-Save")
    trigger_parser.add_argument("--agent", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--repo", default=".")
    list_parser.add_argument("--agent", help="Filter by agent ID")

    args = parser.parse_args()

    if args.command == "trigger":
        trigger_snapshot(args.repo, args.message, args.agent)
    elif args.command == "list":
        list_snapshots(args.repo, args.agent)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
