#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Any, Dict
import subprocess # Import subprocess
import os # Import os

# Ensure nucleus.sdk is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from nucleus.tools.agent_bus import iter_lines_follow, resolve_bus_paths, default_actor
    from nucleus.sdk.planner_executor_schema import RepoExecRequest, RepoExecResult
except ImportError:
    # Fallback/Bootstrap mode for standalone execution
    current_dir = Path(__file__).resolve().parents[0]
    sys.path.append(str(current_dir.parents[0] / "tools")) # For agent_bus
    from agent_bus import iter_lines_follow, resolve_bus_paths, default_actor
    # Minimal stub for schema if SDK not fully installed
    class RepoExecRequest:
        def __init__(self, **kwargs): pass
    class RepoExecResult:
        def __init__(self, **kwargs): pass

def emit_bus_via_subprocess(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data: Dict[str, Any]) -> None:
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        tool = Path(__file__).resolve().parents[0] / "agent_bus.py"
    if not tool.exists():
        return
    
    # Ensure data is a JSON string for subprocess call
    data_json_str = json.dumps(data, ensure_ascii=False)
    print(f"[{actor}] Emitting bus event via subprocess: topic={topic}, kind={kind}, data={data_json_str}")

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--bus-dir",
            bus_dir,
            "pub",
            "--topic",
            topic,
            "--kind",
            kind,
            "--level",
            level,
            "--actor",
            actor,
            "--data",
            data_json_str, # Pass data as a string
        ],
        check=True, # Ensure errors are propagated
        # stdout=subprocess.DEVNULL, # Removed for debugging
        # stderr=subprocess.DEVNULL, # Removed for debugging
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def main():
    bus_paths = resolve_bus_paths(None)
    actor = default_actor()
    
    print(f"[{actor}] Repo Executor Stub: Listening for repo.exec.request events...")
    print(f"[{actor}] Following events at: {bus_paths.events_path}")

    # Use iter_lines_follow for robust tailing
    for line in iter_lines_follow(str(bus_paths.events_path), poll_s=0.1):
        try:
            event = json.loads(line)
            if event.get("topic") == "repo.exec.request" and event.get("kind") == "request":
                req_id = event["data"]["req_id"]
                print(f"[{actor}] Received repo.exec.request for {req_id}. Emitting stubbed response.")

                # Simulate some work / delay
                time.sleep(1) 

                # Emit stubbed response
                result_data = {
                    "req_id": req_id,
                    "status": "stubbed",
                    "commit_sha": None,
                    "artifacts": [],
                    "errors": [f"Execution stubbed by {actor}"],
                }
                emit_bus_via_subprocess(
                    bus_paths.bus_dir,
                    topic="repo.exec.result",
                    kind="response",
                    level="info",
                    actor=actor,
                    data=result_data,
                )
        except Exception as e:
            print(f"[{actor}] Error processing event: {e}")

if __name__ == "__main__":
    main()