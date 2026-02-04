#!/usr/bin/env python3
"""
bootstrap.py - Code Agent Bootstrap Module (Step 51)

PBTSO Phase: SKILL, SEQUESTER

Provides:
- Code Agent initialization and lifecycle management
- A2A bus integration for task dispatch
- Ring level configuration for permission control
- PAIP isolation toggle for safe code execution

Bus Topics:
- a2a.code.bootstrap.start
- a2a.code.bootstrap.complete
- a2a.task.dispatch (subscription)

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# =============================================================================
# Configuration
# =============================================================================

class RingLevel(Enum):
    """Security ring levels for Code Agent permissions."""
    RING_0 = 0  # Kernel/constitutional (read-only analysis)
    RING_1 = 1  # System (config modification)
    RING_2 = 2  # Application (standard code changes)
    RING_3 = 3  # User (sandboxed experiments)


@dataclass
class CodeAgentConfig:
    """Configuration for Code Agent bootstrap."""
    agent_id: str = "code-agent"
    ring_level: int = 2
    working_dir: str = "/pluribus"
    max_edits_per_batch: int = 10
    use_paip_isolation: bool = True
    enable_neural_proposals: bool = True
    supported_languages: List[str] = field(default_factory=lambda: ["python", "typescript", "javascript"])
    style_enforcers: Dict[str, str] = field(default_factory=lambda: {
        "python": "black",
        "typescript": "prettier",
        "javascript": "prettier",
    })
    compile_timeout_s: int = 30
    clone_depth: int = 1  # Shallow clone for PAIP isolation
    heartbeat_interval_s: int = 300

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "agent_id": self.agent_id,
            "ring_level": self.ring_level,
            "working_dir": self.working_dir,
            "max_edits_per_batch": self.max_edits_per_batch,
            "use_paip_isolation": self.use_paip_isolation,
            "enable_neural_proposals": self.enable_neural_proposals,
            "supported_languages": self.supported_languages,
            "style_enforcers": self.style_enforcers,
            "compile_timeout_s": self.compile_timeout_s,
            "clone_depth": self.clone_depth,
            "heartbeat_interval_s": self.heartbeat_interval_s,
        }


# =============================================================================
# Agent Bus (Lightweight Implementation)
# =============================================================================

class AgentBus:
    """
    Lightweight event bus for A2A communication.

    Writes events to ndjson file for cross-agent coordination.
    """

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.handlers: Dict[str, List[Callable]] = {}
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus and return event ID."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(full_event) + "\n")

        # Trigger local handlers
        topic = event.get("topic", "")
        for handler in self.handlers.get(topic, []):
            try:
                handler(full_event)
            except Exception:
                pass  # Don't let handler errors crash the bus

        return event_id

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe handler to topic."""
        if topic not in self.handlers:
            self.handlers[topic] = []
        self.handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable) -> None:
        """Unsubscribe handler from topic."""
        if topic in self.handlers:
            self.handlers[topic] = [h for h in self.handlers[topic] if h != handler]


# =============================================================================
# Task Gate (P/E/L/R/Q)
# =============================================================================

class TaskGate(Enum):
    """P/E/L/R/Q Task Gates for validation."""
    PROPOSE = "P"   # Task proposed - needs approval
    EXECUTE = "E"   # Task approved - executing
    LOG = "L"       # Task complete - logging result
    REVIEW = "R"    # Task under review
    QUEUE = "Q"     # Task queued for later


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VETOED = "vetoed"


@dataclass
class CodeTask:
    """Represents a code modification task."""
    id: str
    description: str
    files: List[str]
    gate: TaskGate
    status: TaskStatus
    created_at: float
    updated_at: float
    proposal_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "files": self.files,
            "gate": self.gate.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "proposal_id": self.proposal_id,
            "result": self.result,
            "error": self.error,
        }


# =============================================================================
# Code Agent Bootstrap
# =============================================================================

class CodeAgentBootstrap:
    """
    Code Agent bootstrap and lifecycle manager.

    PBTSO Phase: SKILL (capability registration), SEQUESTER (isolation setup)

    Responsibilities:
    - Initialize Code Agent with configuration
    - Register with A2A dispatcher
    - Set up PAIP isolation if enabled
    - Manage agent lifecycle (start, heartbeat, shutdown)
    """

    BUS_TOPICS = {
        "bootstrap_start": "a2a.code.bootstrap.start",
        "bootstrap_complete": "a2a.code.bootstrap.complete",
        "task_dispatch": "a2a.task.dispatch",
        "heartbeat": "a2a.code.heartbeat",
        "shutdown": "a2a.code.shutdown",
    }

    def __init__(self, config: Optional[CodeAgentConfig] = None):
        self.config = config or CodeAgentConfig()
        self.bus = AgentBus()
        self.tasks: Dict[str, CodeTask] = {}
        self.is_running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None

        # Component references (set during bootstrap)
        self.proposal_generator = None
        self.edit_coordinator = None
        self.clone_manager = None
        self.style_enforcer = None
        self.incremental_compiler = None

    async def bootstrap(self) -> bool:
        """
        Bootstrap the Code Agent.

        Steps:
        1. Emit bootstrap.start event
        2. Initialize components
        3. Subscribe to task dispatch
        4. Start heartbeat
        5. Emit bootstrap.complete event

        Returns:
            True if bootstrap successful, False otherwise.
        """
        self._start_time = time.time()

        # Emit bootstrap start
        self.bus.emit({
            "topic": self.BUS_TOPICS["bootstrap_start"],
            "kind": "lifecycle",
            "actor": self.config.agent_id,
            "data": {
                "ring": self.config.ring_level,
                "paip": self.config.use_paip_isolation,
                "languages": self.config.supported_languages,
                "working_dir": self.config.working_dir,
            }
        })

        try:
            # Initialize components (lazy imports to avoid circular deps)
            await self._initialize_components()

            # Subscribe to task dispatch
            self.bus.subscribe(
                self.BUS_TOPICS["task_dispatch"],
                self._handle_task_dispatch
            )

            # Start heartbeat
            self.is_running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Emit bootstrap complete
            self.bus.emit({
                "topic": self.BUS_TOPICS["bootstrap_complete"],
                "kind": "lifecycle",
                "actor": self.config.agent_id,
                "data": {
                    "success": True,
                    "components": self._get_component_status(),
                    "elapsed_ms": int((time.time() - self._start_time) * 1000),
                }
            })

            return True

        except Exception as e:
            self.bus.emit({
                "topic": self.BUS_TOPICS["bootstrap_complete"],
                "kind": "lifecycle",
                "level": "error",
                "actor": self.config.agent_id,
                "data": {
                    "success": False,
                    "error": str(e),
                    "elapsed_ms": int((time.time() - self._start_time) * 1000),
                }
            })
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown the Code Agent."""
        self.is_running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Cleanup PAIP clones if manager exists
        if self.clone_manager:
            for task_id in list(self.tasks.keys()):
                try:
                    self.clone_manager.cleanup(task_id)
                except Exception:
                    pass

        self.bus.emit({
            "topic": self.BUS_TOPICS["shutdown"],
            "kind": "lifecycle",
            "actor": self.config.agent_id,
            "data": {
                "tasks_processed": len(self.tasks),
                "uptime_s": int(time.time() - self._start_time) if self._start_time else 0,
            }
        })

    async def _initialize_components(self) -> None:
        """Initialize Code Agent components."""
        from .neural.proposal_generator import NeuralCodeProposalGenerator
        from .edit.coordinator import MultiFileEditCoordinator
        from .paip.clone_manager import PAIPCloneManager
        from .style.enforcer import CodeStyleEnforcer
        from .compile.incremental import IncrementalCompiler

        working_path = Path(self.config.working_dir)

        # Initialize proposal generator
        if self.config.enable_neural_proposals:
            self.proposal_generator = NeuralCodeProposalGenerator(bus=self.bus)

        # Initialize edit coordinator
        self.edit_coordinator = MultiFileEditCoordinator(
            working_dir=working_path,
            bus=self.bus,
        )

        # Initialize PAIP clone manager if isolation enabled
        if self.config.use_paip_isolation:
            self.clone_manager = PAIPCloneManager(
                source_repo=working_path,
                bus=self.bus,
            )

        # Initialize style enforcer
        self.style_enforcer = CodeStyleEnforcer(bus=self.bus)

        # Initialize incremental compiler
        self.incremental_compiler = IncrementalCompiler(
            project_root=working_path,
            bus=self.bus,
        )

    def _get_component_status(self) -> Dict[str, bool]:
        """Get initialization status of components."""
        return {
            "proposal_generator": self.proposal_generator is not None,
            "edit_coordinator": self.edit_coordinator is not None,
            "clone_manager": self.clone_manager is not None,
            "style_enforcer": self.style_enforcer is not None,
            "incremental_compiler": self.incremental_compiler is not None,
        }

    def _handle_task_dispatch(self, event: Dict[str, Any]) -> None:
        """Handle incoming task dispatch events."""
        data = event.get("data", {})
        target = data.get("target")

        # Only handle tasks targeted at this agent
        if target != self.config.agent_id:
            return

        task_id = data.get("task_id", str(uuid.uuid4()))
        now = time.time()

        task = CodeTask(
            id=task_id,
            description=data.get("description", ""),
            files=data.get("files", []),
            gate=TaskGate(data.get("gate", "P")),
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        self.tasks[task_id] = task

        # Emit task accepted
        self.bus.emit({
            "topic": "code.task.accepted",
            "kind": "task",
            "actor": self.config.agent_id,
            "data": task.to_dict(),
        })

    async def _heartbeat_loop(self) -> None:
        """Emit periodic heartbeats."""
        while self.is_running:
            await asyncio.sleep(self.config.heartbeat_interval_s)

            if self.is_running:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["heartbeat"],
                    "kind": "heartbeat",
                    "actor": self.config.agent_id,
                    "data": {
                        "uptime_s": int(time.time() - self._start_time) if self._start_time else 0,
                        "tasks_pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                        "tasks_completed": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
                    }
                })

    # =========================================================================
    # Task Processing
    # =========================================================================

    async def process_task(self, task_id: str) -> Dict[str, Any]:
        """
        Process a code task through the full pipeline.

        PBTSO Phases:
        - PLAN: Generate proposal via neural generator
        - ITERATE: Apply AST transformations
        - VERIFY: Run style enforcement and compilation

        Args:
            task_id: ID of the task to process

        Returns:
            Dict with processing results
        """
        task = self.tasks.get(task_id)
        if not task:
            return {"error": f"Task not found: {task_id}"}

        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = time.time()

        try:
            results: Dict[str, Any] = {"task_id": task_id, "phases": []}

            # PLAN phase: Generate proposal
            if self.proposal_generator and task.gate == TaskGate.PROPOSE:
                proposal = await self.proposal_generator.generate_proposal(
                    task=task.description,
                    context={"files": task.files}
                )
                task.proposal_id = proposal.id
                results["phases"].append({
                    "phase": "PLAN",
                    "proposal_id": proposal.id,
                    "files_affected": proposal.files_affected,
                })

            # ITERATE phase: Apply edits (in PAIP clone if enabled)
            if self.config.use_paip_isolation and self.clone_manager:
                clone_path = self.clone_manager.create_clone(task_id)
                results["phases"].append({
                    "phase": "SEQUESTER",
                    "clone_path": str(clone_path),
                })

            # VERIFY phase: Style and compile
            if self.style_enforcer:
                for file_path in task.files:
                    self.style_enforcer.format_file(Path(file_path))

            if self.incremental_compiler:
                compile_result = self.incremental_compiler.compile_changed()
                results["phases"].append({
                    "phase": "TEST",
                    "compile_result": compile_result,
                })

            task.status = TaskStatus.COMPLETED
            task.result = results
            task.updated_at = time.time()

            self.bus.emit({
                "topic": "code.task.complete",
                "kind": "task",
                "actor": self.config.agent_id,
                "data": task.to_dict(),
            })

            return results

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.updated_at = time.time()

            self.bus.emit({
                "topic": "code.task.failed",
                "kind": "task",
                "level": "error",
                "actor": self.config.agent_id,
                "data": {"task_id": task_id, "error": str(e)},
            })

            return {"error": str(e)}

    def get_task(self, task_id: str) -> Optional[CodeTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def get_pending_tasks(self) -> List[CodeTask]:
        """Get all pending tasks."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Code Agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Agent Bootstrap (Step 51)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start command
    start_parser = subparsers.add_parser("start", help="Start Code Agent")
    start_parser.add_argument("--ring", type=int, default=2, help="Ring level (0-3)")
    start_parser.add_argument("--paip", action="store_true", default=True, help="Enable PAIP isolation")
    start_parser.add_argument("--no-paip", action="store_true", help="Disable PAIP isolation")
    start_parser.add_argument("--working-dir", default="/pluribus", help="Working directory")

    # status command
    subparsers.add_parser("status", help="Show agent status")

    args = parser.parse_args()

    if args.command == "start":
        config = CodeAgentConfig(
            ring_level=args.ring,
            use_paip_isolation=not args.no_paip,
            working_dir=args.working_dir,
        )
        agent = CodeAgentBootstrap(config)

        async def run():
            success = await agent.bootstrap()
            if success:
                print(f"Code Agent started: {config.agent_id}")
                print(f"  Ring Level: {config.ring_level}")
                print(f"  PAIP Isolation: {config.use_paip_isolation}")
                print(f"  Working Dir: {config.working_dir}")
                # In a real deployment, we'd run an event loop here
                await agent.shutdown()
            else:
                print("Failed to start Code Agent")
                return 1
            return 0

        return asyncio.run(run())

    elif args.command == "status":
        print("Code Agent Status: Not running (use 'start' command)")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
