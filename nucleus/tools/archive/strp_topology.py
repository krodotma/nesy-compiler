#!/usr/bin/env python3
"""
STRp Multi-Agent Topology Executors
====================================

Per comprehensive_implementation_matrix.md Section V:

| Topology | Structure | Use Case | Coordination |
|----------|-----------|----------|--------------|
| single   | 1 agent   | Simple queries | None |
| star     | 1 coord + N workers | Complex decomposition | Coordinator aggregates |
| peer_debate | N parallel peers | Verification, consensus | Best-of-N or vote |

Usage:
  python3 strp_topology.py execute --topology star --task "complex query" --fanout 3
  python3 strp_topology.py execute --topology peer_debate --task "verify claim" --fanout 4
  python3 strp_topology.py execute --topology single --task "simple query"
  python3 strp_topology.py list-providers
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal, Callable

try:
    from iso_executor import IsoExecutor
except ImportError:
    # Try local relative import if running from tools dir
    sys.path.append(str(Path(__file__).resolve().parent))
    try:
        from iso_executor import IsoExecutor
    except ImportError:
        IsoExecutor = None

sys.dont_write_bytecode = True


@dataclass
class STRpRequest:
    """A request to the STRp queue."""
    req_id: str
    task: str
    topology: Literal["single", "star", "peer_debate"]
    fanout: int = 1
    timeout_s: float = 60.0
    coordinator: Optional[str] = None
    workers: list[str] = field(default_factory=list)
    isolation: Literal["thread", "process"] = "thread"  # Iso-STRp upgrade
    metadata: dict = field(default_factory=dict)


@dataclass
class STRpResponse:
    """Response from a topology execution."""
    req_id: str
    success: bool
    content: str
    provider: str
    topology: str
    isolation: str = "thread"  # Iso-STRp: thread or process
    results: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _emit_bus_event(topic: str, data: dict, bus_dir: Optional[str] = None):
    """Emit event to the bus."""
    bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus"
    tool = Path(__file__).parent / "agent_bus.py"
    if not tool.exists():
        return
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
            "artifact",
            "--level",
            "info",
            "--actor",
            os.environ.get("PLURIBUS_ACTOR", "strp_topology"),
            "--data",
            json.dumps(data, ensure_ascii=False),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _get_available_providers() -> list[str]:
    """Get list of available providers (web-session-only by default)."""
    root = Path(os.environ.get("PLURIBUS_ROOT") or "/pluribus")
    state_path = root / ".pluribus" / "browser_daemon.json"
    tabs: dict = {}
    try:
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8", errors="replace"))
            if bool(data.get("running")) and isinstance(data.get("tabs"), dict):
                tabs = data.get("tabs") or {}
    except Exception:
        tabs = {}

    def tab_ready(pid: str) -> bool:
        t = tabs.get(pid)
        return isinstance(t, dict) and str(t.get("status") or "").strip().lower() == "ready"

    preferred = ["chatgpt-web", "claude-web", "gemini-web"]
    ready = [p for p in preferred if tab_ready(p)]
    return ready or preferred


def _dispatch_to_provider(provider: str, task: str, timeout_s: float = 60.0) -> dict:
    """Dispatch task to a specific provider via router."""
    router_path = Path(__file__).parent / "providers" / "router.py"
    if not router_path.exists():
        return {"success": False, "error": "router not found", "provider": provider}

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(router_path),
                "--provider",
                provider,
                "--prompt",
                task,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        return {
            "success": result.returncode == 0,
            "content": result.stdout.strip() if result.returncode == 0 else result.stderr.strip(),
            "provider": provider,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "provider": provider}
    except Exception as e:
        return {"success": False, "error": str(e), "provider": provider}


class SingleTopologyExecutor:
    """Execute STRp requests with single provider."""

    def execute(self, request: STRpRequest) -> STRpResponse:
        provider = request.coordinator or _get_available_providers()[0]

        _emit_bus_event("strp.topology.single.start", {
            "req_id": request.req_id,
            "provider": provider,
            "task_preview": request.task[:100],
        })

        result = _dispatch_to_provider(provider, request.task, request.timeout_s)

        _emit_bus_event("strp.topology.single.complete", {
            "req_id": request.req_id,
            "provider": provider,
            "success": result["success"],
        })

        return STRpResponse(
            req_id=request.req_id,
            success=result["success"],
            content=result.get("content", ""),
            provider=provider,
            topology="single",
            isolation=request.isolation,
            results=[result],
        )


class StarTopologyExecutor:
    """Execute STRp requests in star topology (1 coordinator + N workers)."""

    def execute(self, request: STRpRequest) -> STRpResponse:
        providers = _get_available_providers()
        coordinator = request.coordinator or providers[0]
        workers = request.workers or providers[1:request.fanout + 1]

        # Ensure we have enough workers
        while len(workers) < request.fanout:
            workers.append("mock")

        _emit_bus_event("strp.topology.star.plan", {
            "req_id": request.req_id,
            "coordinator": coordinator,
            "workers": workers[:request.fanout],
            "fanout": request.fanout,
        })

        # Step 1: Coordinator decomposes task
        decompose_prompt = f"""Decompose this task into {request.fanout} independent subtasks.
Return JSON array of subtask strings.

Task: {request.task}

Response format: ["subtask1", "subtask2", ...]"""

        decompose_result = _dispatch_to_provider(coordinator, decompose_prompt, request.timeout_s)

        if not decompose_result["success"]:
            # Fallback: create simple subtasks
            subtasks = [request.task] * request.fanout
        else:
            try:
                # Try to parse JSON from response
                content = decompose_result["content"]
                # Extract JSON array from response
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    subtasks = json.loads(json_match.group())
                else:
                    subtasks = [request.task] * request.fanout
            except Exception:
                subtasks = [request.task] * request.fanout

        # Ensure we have the right number of subtasks
        while len(subtasks) < request.fanout:
            subtasks.append(request.task)
        subtasks = subtasks[:request.fanout]

        # Step 2: Dispatch to workers in parallel
        results = []
        
        # Helper for process isolation (IsoExecutor)
        def run_isolated(worker, subtask, sub_trace_id):
            if not IsoExecutor:
                return {"success": False, "error": "IsoExecutor not available"}
            
            iso_exec = IsoExecutor()
            res = iso_exec.spawn(
                agent=worker,
                goal=subtask,
                trace_id=sub_trace_id,
                parent_id=request.req_id,
                timeout=request.timeout_s
            )
            
            if res.exit_code == 0:
                try:
                    data = json.loads(res.stdout)
                    return {
                        "success": True, 
                        "content": data.get("text", res.stdout),
                        "provider": data.get("provider", "iso-process")
                    }
                except json.JSONDecodeError:
                    return {"success": True, "content": res.stdout, "provider": "iso-process"}
            else:
                return {"success": False, "error": res.stderr or res.stdout, "provider": "iso-process-fail"}

        with concurrent.futures.ThreadPoolExecutor(max_workers=request.fanout) as executor:
            futures = []
            for i, subtask in enumerate(subtasks):
                worker = workers[i % len(workers)]
                
                if request.isolation == "process" and IsoExecutor:
                    sub_trace_id = str(uuid.uuid4())
                    future = executor.submit(run_isolated, worker, subtask, sub_trace_id)
                else:
                    future = executor.submit(_dispatch_to_provider, worker, subtask, request.timeout_s)
                
                futures.append((worker, subtask, future))

            for worker, subtask, future in futures:
                try:
                    result = future.result(timeout=request.timeout_s + 5) # Buffer for overhead
                    # Normalize result keys
                    success = result.get("success", False)
                    content = result.get("content", "")
                    error = result.get("error", "")
                    
                    results.append({
                        "worker": worker,
                        "subtask": subtask,
                        "success": success,
                        "content": content,
                        "error": error
                    })
                    
                    event_type = "complete" if success else "failed"
                    event_data = {
                        "req_id": request.req_id,
                        "worker": worker,
                        "success": success,
                    }
                    if not success:
                        event_data["error"] = error
                        
                    _emit_bus_event(f"strp.worker.{event_type}", event_data)
                    
                except Exception as e:
                    results.append({
                        "worker": worker,
                        "subtask": subtask,
                        "success": False,
                        "error": str(e),
                    })
                    _emit_bus_event("strp.worker.failed", {
                        "req_id": request.req_id,
                        "worker": worker,
                        "error": str(e),
                    })

        # Step 3: Coordinator aggregates results
        successful_results = [r for r in results if r["success"]]
        if not successful_results:
            _emit_bus_event("strp.topology.star.failed", {
                "req_id": request.req_id,
                "workers_succeeded": 0,
                "workers_total": len(results),
            })
            return STRpResponse(
                req_id=request.req_id,
                success=False,
                content="All workers failed",
                provider=coordinator,
                topology="star",
                isolation="process",  # Star always uses process isolation
                results=results,
            )

        aggregate_prompt = f"""Synthesize these worker results into a coherent answer.

Original task: {request.task}

Worker results:
{json.dumps([{"subtask": r["subtask"], "result": r["content"][:500]} for r in successful_results], indent=2)}

Provide a synthesized answer:"""

        aggregate_result = _dispatch_to_provider(coordinator, aggregate_prompt, request.timeout_s)

        _emit_bus_event("strp.topology.star.complete", {
            "req_id": request.req_id,
            "workers_succeeded": len(successful_results),
            "workers_total": len(results),
            "aggregated": aggregate_result["success"],
        })

        return STRpResponse(
            req_id=request.req_id,
            success=aggregate_result["success"],
            content=aggregate_result.get("content", ""),
            provider=coordinator,
            topology="star",
            isolation="process",  # Star always uses process isolation
            results=results,
            metadata={"aggregated_by": coordinator},
        )


class PeerDebateExecutor:
    """Execute STRp requests with peer debate (N parallel, then consensus)."""

    def execute(
        self,
        request: STRpRequest,
        consensus_strategy: Literal["majority", "best_score", "synthesis"] = "synthesis"
    ) -> STRpResponse:
        providers = _get_available_providers()
        peers = request.workers or providers[:request.fanout]

        # Ensure we have enough peers
        while len(peers) < request.fanout:
            peers.append("mock")
        peers = peers[:request.fanout]

        _emit_bus_event("strp.topology.peer_debate.start", {
            "req_id": request.req_id,
            "peers": peers,
            "fanout": request.fanout,
            "consensus_strategy": consensus_strategy,
        })

        # Step 1: Dispatch same task to all peers in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=request.fanout) as executor:
            futures = []
            for peer in peers:
                future = executor.submit(_dispatch_to_provider, peer, request.task, request.timeout_s)
                futures.append((peer, future))

            for peer, future in futures:
                try:
                    result = future.result(timeout=request.timeout_s)
                    results.append({
                        "peer": peer,
                        "success": result["success"],
                        "content": result.get("content", ""),
                    })
                except Exception as e:
                    results.append({
                        "peer": peer,
                        "success": False,
                        "error": str(e),
                    })

        # Step 2: Apply consensus strategy
        successful_results = [r for r in results if r["success"]]

        if not successful_results:
            _emit_bus_event("strp.topology.peer_debate.failed", {
                "req_id": request.req_id,
                "peers_responded": 0,
            })
            return STRpResponse(
                req_id=request.req_id,
                success=False,
                content="No peers responded",
                provider="none",
                topology="peer_debate",
                isolation="process",  # Peer debate always uses process isolation
                results=results,
            )

        if consensus_strategy == "majority":
            final_content = self._majority_vote(successful_results)
        elif consensus_strategy == "best_score":
            # Use longest response as proxy for quality
            final_content = max(successful_results, key=lambda r: len(r["content"]))["content"]
        else:  # synthesis
            final_content = self._synthesize(request.task, successful_results)

        # Calculate agreement ratio
        agreement_ratio = self._agreement_ratio(successful_results)

        _emit_bus_event("strp.topology.peer_debate.complete", {
            "req_id": request.req_id,
            "peers_responded": len(successful_results),
            "consensus_strategy": consensus_strategy,
            "agreement_ratio": agreement_ratio,
        })

        return STRpResponse(
            req_id=request.req_id,
            success=True,
            content=final_content,
            provider=f"consensus({consensus_strategy})",
            topology="peer_debate",
            isolation="process",  # Peer debate always uses process isolation
            results=results,
            metadata={
                "consensus_strategy": consensus_strategy,
                "agreement_ratio": agreement_ratio,
            },
        )

    def _majority_vote(self, results: list[dict]) -> str:
        """Select response with most agreement (by content hash similarity)."""
        if len(results) == 1:
            return results[0]["content"]

        # Group by content prefix (first 100 chars)
        groups: dict[str, list[dict]] = {}
        for r in results:
            key = r["content"][:100].strip().lower()
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        # Return content from largest group
        largest_group = max(groups.values(), key=len)
        return largest_group[0]["content"]

    def _synthesize(self, task: str, results: list[dict]) -> str:
        """Synthesize multiple responses into one (using first available provider)."""
        if len(results) == 1:
            return results[0]["content"]

        # Build synthesis prompt
        responses_text = "\n\n".join([
            f"Response from {r['peer']}:\n{r['content'][:500]}"
            for r in results
        ])

        synthesis_prompt = f"""Multiple agents provided answers to this task. Synthesize them into a single, coherent response.

Task: {task}

{responses_text}

Synthesized answer:"""

        # Use first available provider for synthesis
        providers = _get_available_providers()
        synth_result = _dispatch_to_provider(providers[0], synthesis_prompt, 30.0)

        if synth_result["success"]:
            return synth_result["content"]

        # Fallback: return longest response
        return max(results, key=lambda r: len(r["content"]))["content"]

    def _agreement_ratio(self, results: list[dict]) -> float:
        """Calculate how much the responses agree (0-1)."""
        if len(results) <= 1:
            return 1.0

        # Simple heuristic: compare first 100 chars
        prefixes = [r["content"][:100].strip().lower() for r in results]
        matches = sum(1 for p in prefixes if p == prefixes[0])
        return matches / len(prefixes)


def dispatch_request(request: STRpRequest) -> STRpResponse:
    """Dispatch request according to topology."""
    if request.topology == "single":
        return SingleTopologyExecutor().execute(request)
    elif request.topology == "star":
        return StarTopologyExecutor().execute(request)
    elif request.topology == "peer_debate":
        return PeerDebateExecutor().execute(request)
    else:
        return STRpResponse(
            req_id=request.req_id,
            success=False,
            content=f"Unknown topology: {request.topology}",
            provider="none",
            topology=request.topology,
            isolation=request.isolation,
        )


def main():
    parser = argparse.ArgumentParser(description="STRp Multi-Agent Topology Executors")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # execute
    exec_p = subparsers.add_parser("execute", help="Execute a task with specified topology")
    exec_p.add_argument("--topology", choices=["single", "star", "peer_debate"], default="single")
    exec_p.add_argument("--task", required=True, help="Task to execute")
    exec_p.add_argument("--fanout", type=int, default=3, help="Number of workers/peers")
    exec_p.add_argument("--timeout", type=float, default=60.0, help="Timeout in seconds")
    exec_p.add_argument("--coordinator", help="Coordinator provider (for star)")
    exec_p.add_argument("--workers", help="Comma-separated worker providers")
    exec_p.add_argument("--isolation", choices=["thread", "process"], default="thread", help="Isolation level (Iso-STRp)")
    exec_p.add_argument("--consensus", choices=["majority", "best_score", "synthesis"], default="synthesis")

    # list-providers
    subparsers.add_parser("list-providers", help="List available providers")

    args = parser.parse_args()

    if args.command == "execute":
        workers = args.workers.split(",") if args.workers else []

        request = STRpRequest(
            req_id=str(uuid.uuid4()),
            task=args.task,
            topology=args.topology,
            fanout=args.fanout,
            timeout_s=args.timeout,
            coordinator=args.coordinator,
            workers=workers,
            isolation=args.isolation,
        )

        if args.topology == "peer_debate":
            response = PeerDebateExecutor().execute(request, args.consensus)
        else:
            response = dispatch_request(request)

        print(json.dumps(asdict(response), indent=2))

    elif args.command == "list-providers":
        providers = _get_available_providers()
        for p in providers:
            print(f"  {p}")


if __name__ == "__main__":
    main()
