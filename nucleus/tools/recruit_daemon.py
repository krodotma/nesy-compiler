#!/usr/bin/env python3
"""
recruit_daemon.py - Automatic Agent Recruitment Daemon

Listens to bus for swarm.recruit.request events and:
1. First notifies existing agents via HEXIS buffers
2. Waits 2 minutes for acknowledgements
3. If no response, spawns new persistent agent sessions

Spawned agents are persistent CLIs (not one-off prompts).

Usage:
    python3 recruit_daemon.py --daemon              # Run as daemon
    python3 recruit_daemon.py --once                # Process once and exit
    python3 recruit_daemon.py --check-pending       # Show pending recruitments

Ring: 0 (Infrastructure)
Protocol: DKIN v28 | PAIP v15 | Citizen v1
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from collections import deque
import threading


# Configuration
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))
LEDGER_DIR = Path(os.environ.get("PLURIBUS_DKIN_DIR", ".pluribus/dkin"))
HEXIS_DIR = Path("/tmp")
POLL_INTERVAL_S = 5
ACK_TIMEOUT_S = 120  # 2 minutes before fallback spawn
SPAWN_MODELS = {
    "claude": "claude",
    "codex": "codex", 
    "gemini": "gemini",
    "qwen": "qwen"
}


@dataclass
class PendingRecruitment:
    """Tracks a pending recruitment request."""
    recruitment_id: str
    task: str
    priority: str
    requested_agents: List[str]
    acked_agents: List[str] = field(default_factory=list)
    spawned_agents: List[str] = field(default_factory=list)
    requested_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, acked, spawned, complete


class RecruitDaemon:
    """
    Daemon that listens for recruitment requests and manages agent lifecycle.
    """
    
    def __init__(self, bus_dir: Path = None, working_dir: str = None):
        self.bus_dir = bus_dir or BUS_DIR
        self.bus_path = self.bus_dir / "events.ndjson"
        self.ledger_path = LEDGER_DIR / "recruit_daemon.ndjson"
        self.working_dir = working_dir or os.getcwd()
        
        # State
        self.pending: Dict[str, PendingRecruitment] = {}
        self.last_event_ts = 0.0
        self.running = False
        
        self._load_state()
    
    def _load_state(self):
        """Load pending recruitments from ledger."""
        if self.ledger_path.exists():
            try:
                for line in self.ledger_path.read_text().strip().split('\n')[-50:]:
                    if line:
                        data = json.loads(line)
                        if data.get("type") == "pending" and data.get("status") == "pending":
                            rec = PendingRecruitment(**{k: v for k, v in data.items() 
                                                        if k in PendingRecruitment.__dataclass_fields__})
                            self.pending[rec.recruitment_id] = rec
            except Exception as e:
                print(f"⚠️ Error loading state: {e}")
    
    def _save_recruitment(self, rec: PendingRecruitment):
        """Save recruitment state to ledger."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"type": "pending", "ts": time.time(), **asdict(rec)}
        with open(self.ledger_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def _emit_bus_event(self, topic: str, data: Dict[str, Any], level: str = "info"):
        """Emit event to bus."""
        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "kind": "event",
            "level": level,
            "actor": "recruit_daemon",
            "data": data
        }
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bus_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def _check_hexis_ack(self, agent: str, recruitment_id: str) -> bool:
        """Check if agent has acknowledged via HEXIS response."""
        ack_path = HEXIS_DIR / f"{agent}.ack"
        if ack_path.exists():
            try:
                content = ack_path.read_text()
                if recruitment_id in content:
                    return True
            except Exception:
                pass
        return False
    
    def _spawn_persistent_agent(self, agent_id: str, task: str, model: str = None) -> bool:
        """
        Spawn a persistent CLI agent session in tmux.
        
        Unlike one-off prompts, this creates an interactive session that
        stays alive until explicitly killed.
        """
        model = model or SPAWN_MODELS.get(agent_id, "claude")
        session_name = f"pluribus_agent_{agent_id}"
        
        # Check if session already exists
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True
        )
        if result.returncode == 0:
            print(f"[Spawn] Session {session_name} already exists, reusing")
            # Send task to existing session
            subprocess.run([
                "tmux", "send-keys", "-t", session_name,
                f"# New task: {task}", "Enter"
            ])
            return True
        
        # Create new persistent session
        try:
            # Create detached session
            subprocess.run([
                "tmux", "new-session", "-d", "-s", session_name,
                "-c", self.working_dir
            ], check=True)
            
            # Start the CLI in interactive mode (not -p one-off)
            cli_cmd = f"{model}"  # Just the CLI, no -p flag = interactive
            subprocess.run([
                "tmux", "send-keys", "-t", session_name,
                f"cd {self.working_dir} && {cli_cmd}", "Enter"
            ], check=True)
            
            # Wait for CLI to start
            time.sleep(2)
            
            # Send initial task
            subprocess.run([
                "tmux", "send-keys", "-t", session_name,
                task, "Enter"
            ], check=True)
            
            print(f"[Spawn] Created persistent session: {session_name}")
            return True
            
        except Exception as e:
            print(f"[Spawn] Error spawning {agent_id}: {e}")
            return False
    
    def process_recruitment_request(self, event: Dict[str, Any]):
        """Process a swarm.recruit.request event."""
        data = event.get("data", {})
        recruitment_id = data.get("recruitment_id", uuid.uuid4().hex[:8])
        task = data.get("task", "No task specified")
        agents = data.get("agents", ["claude", "codex", "gemini", "qwen", "grok"])
        priority = data.get("priority", "P1")
        
        print(f"[Recruit] New request: {recruitment_id}")
        print(f"[Recruit] Task: {task}")
        print(f"[Recruit] Agents: {agents}")
        
        # Create pending recruitment
        rec = PendingRecruitment(
            recruitment_id=recruitment_id,
            task=task,
            priority=priority,
            requested_agents=agents
        )
        self.pending[recruitment_id] = rec
        self._save_recruitment(rec)
        
        # Notify agents via HEXIS (already done by pbrecruit/pbnotify)
        # We just wait for acks
        self._emit_bus_event("recruit.daemon.waiting", {
            "recruitment_id": recruitment_id,
            "waiting_for": agents,
            "timeout_s": ACK_TIMEOUT_S
        })
    
    def process_ack(self, event: Dict[str, Any]):
        """Process an agent.notify.ack event."""
        data = event.get("data", {})
        req_id = data.get("req_id", "")
        actor = event.get("actor", data.get("actor", ""))
        
        # Find matching recruitment
        for rec in self.pending.values():
            if req_id in rec.recruitment_id or rec.recruitment_id in str(data):
                if actor not in rec.acked_agents:
                    rec.acked_agents.append(actor)
                    print(f"[Recruit] ACK from {actor} for {rec.recruitment_id}")
                    self._save_recruitment(rec)
                break
    
    def check_timeouts(self):
        """Check for recruitments that have timed out and need spawn."""
        now = time.time()
        
        for rec_id, rec in list(self.pending.items()):
            if rec.status != "pending":
                continue
            
            elapsed = now - rec.requested_at
            
            # Check which agents haven't acked
            unacked = [a for a in rec.requested_agents if a not in rec.acked_agents]
            
            if elapsed >= ACK_TIMEOUT_S and unacked:
                print(f"[Recruit] Timeout for {rec_id}, spawning {len(unacked)} agents")
                
                for agent in unacked:
                    if agent in SPAWN_MODELS:
                        success = self._spawn_persistent_agent(agent, rec.task)
                        if success:
                            rec.spawned_agents.append(agent)
                
                rec.status = "spawned" if rec.spawned_agents else "timeout_failed"
                self._save_recruitment(rec)
                
                self._emit_bus_event("recruit.daemon.spawned", {
                    "recruitment_id": rec_id,
                    "spawned": rec.spawned_agents,
                    "failed": [a for a in unacked if a not in rec.spawned_agents]
                })
            
            # Mark complete if all agents acked
            elif not unacked:
                rec.status = "complete"
                self._save_recruitment(rec)
                print(f"[Recruit] {rec_id} complete - all agents acked")
    
    def poll_bus(self):
        """Poll bus for new events."""
        if not self.bus_path.exists():
            return
        
        try:
            with open(self.bus_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        ts = event.get("ts", 0)
                        
                        if ts <= self.last_event_ts:
                            continue
                        
                        self.last_event_ts = ts
                        topic = event.get("topic", "")
                        
                        if topic == "swarm.recruit.request":
                            self.process_recruitment_request(event)
                        elif topic == "agent.notify.ack":
                            self.process_ack(event)
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[Poll] Error: {e}")
    
    def run_daemon(self):
        """Run as continuous daemon."""
        print(f"[Daemon] Starting recruit daemon")
        print(f"[Daemon] Bus: {self.bus_path}")
        print(f"[Daemon] ACK timeout: {ACK_TIMEOUT_S}s")
        
        self.running = True
        
        while self.running:
            try:
                self.poll_bus()
                self.check_timeouts()
                time.sleep(POLL_INTERVAL_S)
            except KeyboardInterrupt:
                print("\n[Daemon] Shutting down...")
                self.running = False
            except Exception as e:
                print(f"[Daemon] Error: {e}")
                time.sleep(POLL_INTERVAL_S)
    
    def run_once(self):
        """Process once and exit."""
        self.poll_bus()
        self.check_timeouts()
        print(f"[Once] Processed {len(self.pending)} pending recruitments")


def main():
    parser = argparse.ArgumentParser(description="Recruit Daemon - Automatic agent recruitment")
    parser.add_argument("--daemon", "-d", action="store_true", help="Run as daemon")
    parser.add_argument("--once", action="store_true", help="Process once and exit")
    parser.add_argument("--check-pending", action="store_true", help="Show pending recruitments")
    parser.add_argument("--bus-dir", help="Bus directory path")
    parser.add_argument("--working-dir", "-w", help="Working directory for spawned agents")
    
    args = parser.parse_args()
    
    daemon = RecruitDaemon(
        bus_dir=Path(args.bus_dir) if args.bus_dir else None,
        working_dir=args.working_dir
    )
    
    if args.check_pending:
        print(f"Pending recruitments: {len(daemon.pending)}")
        for rec_id, rec in daemon.pending.items():
            print(f"  {rec_id}: {rec.status} - acked: {rec.acked_agents}")
    elif args.once:
        daemon.run_once()
    elif args.daemon:
        daemon.run_daemon()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
