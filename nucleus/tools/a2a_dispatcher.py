#!/usr/bin/env python3
"""
a2a_dispatcher.py - Agent-to-Agent Star Topology Dispatcher

Implements the A2A coordination protocol for multi-agent swarm orchestration:
- Star Topology: Central orchestrator dispatches to subagents
- P/E/L/R/Q Gates: Propose/Execute/Log/Review/Queue validation
- Task Ledger integration for full audit trail

Part of Evolution Phase 2: CAGENT Bootstrap
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Callable, Any

from .dkin_ledger import ledger as dkin_ledger
from .agent_bus import AgentBus

# ============================================================================
# Types
# ============================================================================

class TaskGate(Enum):
    """P/E/L/R/Q Task Gates for validation."""
    PROPOSE = "P"    # Task proposed - needs approval
    EXECUTE = "E"    # Task approved - executing
    LOG = "L"        # Task complete - logging result
    REVIEW = "R"     # Task under review
    QUEUE = "Q"      # Task queued for later

class TaskStatus(Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VETOED = "vetoed"

@dataclass
class A2ATask:
    """Represents a task in the A2A coordination system."""
    id: str
    topic: str
    actor: str           # Dispatcher agent
    target: str          # Target subagent
    gate: TaskGate
    status: TaskStatus
    payload: Dict[str, Any]
    created_at: float
    updated_at: float
    result: Optional[Dict[str, Any]] = None
    veto_reason: Optional[str] = None

# ============================================================================
# A2A Dispatcher (Star Topology Orchestrator)
# ============================================================================

class A2ADispatcher:
    """
    Central orchestrator for Star Topology A2A coordination.
    
    Responsibilities:
    - Dispatch tasks to subagents via bus
    - Track task state through P/E/L/R/Q gates
    - Persist all transitions to DKIN ledger
    - Handle Omega veto (single veto blocks proposal)
    """
    
    BUS_TOPICS = {
        "dispatch": "a2a.task.dispatch",
        "complete": "a2a.task.complete",
        "veto": "a2a.task.veto",
        "progress": "a2a.task.progress",
    }
    
    def __init__(self, actor_id: str = "orchestrator"):
        self.actor_id = actor_id
        self.bus = AgentBus()
        self.tasks: Dict[str, A2ATask] = {}
        self.handlers: Dict[str, List[Callable]] = {}
        
    # ========================================================================
    # Core Dispatch
    # ========================================================================
    
    def dispatch(
        self,
        target: str,
        topic: str,
        payload: Dict[str, Any],
        gate: TaskGate = TaskGate.PROPOSE
    ) -> str:
        """
        Dispatch a task to a target subagent.
        Returns the task ID for tracking.
        """
        task_id = str(uuid.uuid4())
        now = time.time()
        
        task = A2ATask(
            id=task_id,
            topic=topic,
            actor=self.actor_id,
            target=target,
            gate=gate,
            status=TaskStatus.DISPATCHED,
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        
        self.tasks[task_id] = task
        
        # Record to ledger
        self._ledger_record("task_dispatch", task)
        
        # Emit to bus
        self.bus.emit({
            "topic": self.BUS_TOPICS["dispatch"],
            "kind": "a2a_task",
            "actor": self.actor_id,
            "data": {
                "task_id": task_id,
                "target": target,
                "topic": topic,
                "gate": gate.value,
                "payload": payload,
            }
        })
        
        return task_id
    
    def complete(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed with result."""
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.gate = TaskGate.LOG
        task.result = result
        task.updated_at = time.time()
        
        self._ledger_record("task_complete", task)
        
        self.bus.emit({
            "topic": self.BUS_TOPICS["complete"],
            "kind": "a2a_task",
            "actor": self.actor_id,
            "data": {
                "task_id": task_id,
                "result": result,
            }
        })
        
        return True
    
    def veto(self, task_id: str, reason: str) -> bool:
        """
        Veto a task (Geometric Omega - single veto blocks).
        Used for Ring 0 constitutional enforcement.
        """
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        task.status = TaskStatus.VETOED
        task.veto_reason = reason
        task.updated_at = time.time()
        
        self._ledger_record("task_veto", task)
        
        self.bus.emit({
            "topic": self.BUS_TOPICS["veto"],
            "kind": "a2a_task",
            "actor": self.actor_id,
            "data": {
                "task_id": task_id,
                "reason": reason,
            }
        })
        
        return True
    
    # ========================================================================
    # Gate Transitions
    # ========================================================================
    
    def transition_gate(self, task_id: str, new_gate: TaskGate) -> bool:
        """
        Transition a task through P/E/L/R/Q gates.
        Validates allowed transitions.
        """
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        old_gate = task.gate
        
        # Validate transition (simplified - extend as needed)
        allowed = self._validate_gate_transition(old_gate, new_gate)
        if not allowed:
            return False
            
        task.gate = new_gate
        task.updated_at = time.time()
        
        # Update status based on gate
        if new_gate == TaskGate.EXECUTE:
            task.status = TaskStatus.IN_PROGRESS
        elif new_gate == TaskGate.LOG:
            task.status = TaskStatus.COMPLETED
        elif new_gate == TaskGate.QUEUE:
            task.status = TaskStatus.PENDING
            
        self._ledger_record("gate_transition", task, {"from": old_gate.value, "to": new_gate.value})
        
        return True
    
    def _validate_gate_transition(self, from_gate: TaskGate, to_gate: TaskGate) -> bool:
        """Validate P/E/L/R/Q gate transition rules."""
        transitions = {
            TaskGate.PROPOSE: [TaskGate.EXECUTE, TaskGate.QUEUE],
            TaskGate.EXECUTE: [TaskGate.LOG, TaskGate.REVIEW],
            TaskGate.LOG: [],  # Terminal
            TaskGate.REVIEW: [TaskGate.EXECUTE, TaskGate.QUEUE],
            TaskGate.QUEUE: [TaskGate.PROPOSE],
        }
        return to_gate in transitions.get(from_gate, [])
    
    # ========================================================================
    # Ledger Integration
    # ========================================================================
    
    def _ledger_record(self, event_type: str, task: A2ATask, extra: Optional[Dict] = None):
        """Record task event to DKIN ledger."""
        data = {
            "task_id": task.id,
            "actor": task.actor,
            "target": task.target,
            "gate": task.gate.value,
            "status": task.status.value,
        }
        if extra:
            data.update(extra)
            
        dkin_ledger.record(f"a2a.{event_type}", data)
    
    # ========================================================================
    # Query
    # ========================================================================
    
    def get_task(self, task_id: str) -> Optional[A2ATask]:
        return self.tasks.get(task_id)
    
    def get_tasks_by_target(self, target: str) -> List[A2ATask]:
        return [t for t in self.tasks.values() if t.target == target]
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[A2ATask]:
        return [t for t in self.tasks.values() if t.status == status]
    
    def get_pending_count(self) -> int:
        return len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])

# ============================================================================
# Singleton
# ============================================================================

dispatcher = A2ADispatcher()

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="A2A Dispatcher CLI")
    parser.add_argument("--dispatch", metavar="TARGET", help="Dispatch task to target agent")
    parser.add_argument("--topic", default="test.task", help="Task topic")
    parser.add_argument("--payload", default="{}", help="JSON payload")
    parser.add_argument("--status", action="store_true", help="Show dispatcher status")
    
    args = parser.parse_args()
    
    if args.dispatch:
        payload = json.loads(args.payload)
        task_id = dispatcher.dispatch(args.dispatch, args.topic, payload)
        print(f"✅ Dispatched task {task_id} to {args.dispatch}")
        
    if args.status:
        print(f"📊 A2A Dispatcher Status")
        print(f"   Tasks: {len(dispatcher.tasks)}")
        print(f"   Pending: {dispatcher.get_pending_count()}")
