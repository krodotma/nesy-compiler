"""
Theia Swarm Dispatcher — Prescient routing with failover.

Routes A2A requests to healthy providers with automatic failover.
Implements DKIN v18 resilience patterns.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

# Provider priority (prefer local/fast, fallback to cloud)
PROVIDER_PRIORITY = ["codex", "ollama", "claude", "gemini", "aider"]

# Health cache
_provider_health: Dict[str, Dict[str, Any]] = {}
_last_health_check = 0.0


@dataclass
class DispatchRequest:
    """Request to dispatch to swarm."""
    id: str
    topic: str
    payload: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 5
    target_provider: Optional[str] = None


class SwarmDispatcher:
    """
    Prescient dispatcher with automatic failover.
    
    Features:
    - Priority-based provider selection
    - Health-aware routing
    - Automatic retry with backoff
    - Task lifecycle event emission
    """
    
    def __init__(self, bus_path: str = "/pluribus/.pluribus/bus/events.ndjson"):
        self.bus_path = bus_path
        self.pending: List[DispatchRequest] = []
        self.completed: List[str] = []
        self._callbacks: Dict[str, Callable] = {}
    
    def update_health(self, health_data: Dict[str, Dict[str, Any]]) -> None:
        """Update provider health cache."""
        global _provider_health, _last_health_check
        _provider_health = health_data
        _last_health_check = time.time()
    
    def is_healthy(self, provider: str) -> bool:
        """Check if provider is healthy."""
        if provider not in _provider_health:
            return False
        return _provider_health[provider].get("healthy", False)
    
    def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers in priority order."""
        return [p for p in PROVIDER_PRIORITY if self.is_healthy(p)]
    
    def select_provider(self, preferred: Optional[str] = None) -> Optional[str]:
        """Select best available provider."""
        if preferred and self.is_healthy(preferred):
            return preferred
        
        healthy = self.get_healthy_providers()
        return healthy[0] if healthy else None
    
    def dispatch(self, request: DispatchRequest) -> bool:
        """
        Dispatch request to appropriate provider.
        
        Returns True if dispatched, False if queued for retry.
        """
        request.attempts += 1
        
        provider = self.select_provider(request.target_provider)
        
        if not provider:
            if request.attempts < request.max_attempts:
                self.pending.append(request)
                return False
            else:
                self._emit_failure(request, "No healthy providers")
                return False
        
        # Emit dispatch event
        self._emit_event({
            "topic": f"theia.swarm.dispatch",
            "kind": "dispatch",
            "level": "info",
            "actor": "theia-dispatcher",
            "data": {
                "request_id": request.id,
                "provider": provider,
                "attempt": request.attempts,
                "original_topic": request.topic,
            }
        })
        
        # Mark as completed (actual execution would be async)
        self.completed.append(request.id)
        return True
    
    def process_pending(self) -> int:
        """Process pending requests. Returns count processed."""
        processed = 0
        remaining = []
        
        for request in self.pending:
            if self.dispatch(request):
                processed += 1
            else:
                remaining.append(request)
        
        self.pending = remaining
        return processed
    
    def _emit_event(self, event: Dict[str, Any]) -> None:
        """Emit event to bus."""
        event["id"] = str(uuid.uuid4())
        event["ts"] = time.time()
        event["iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[Dispatcher] Failed to emit: {e}")
    
    def _emit_failure(self, request: DispatchRequest, reason: str) -> None:
        """Emit failure event."""
        self._emit_event({
            "topic": "theia.swarm.dispatch.failed",
            "kind": "error",
            "level": "error",
            "actor": "theia-dispatcher",
            "data": {
                "request_id": request.id,
                "reason": reason,
                "attempts": request.attempts,
            }
        })
    
    def create_a2a_request(
        self,
        title: str,
        goals: List[str],
        preferred_provider: Optional[str] = None,
    ) -> DispatchRequest:
        """Create A2A collaboration request."""
        return DispatchRequest(
            id=f"theia-{int(time.time())}-{uuid.uuid4().hex[:8]}",
            topic="a2a.collaboration.request",
            payload={
                "title": title,
                "goals": goals,
                "source": "theia",
                "protocol": "DKIN-v30",
            },
            target_provider=preferred_provider,
        )
    
    def status(self) -> Dict[str, Any]:
        """Get dispatcher status."""
        return {
            "pending": len(self.pending),
            "completed": len(self.completed),
            "healthy_providers": self.get_healthy_providers(),
            "last_health_check": _last_health_check,
        }


# Task lifecycle events (DKIN v18)
class TaskLifecycle:
    """
    DKIN v18 compliant task lifecycle manager.
    
    Emits structured task events for Agentic State Graph.
    """
    
    def __init__(self, agent_id: str = "theia", bus_path: str = "/pluribus/.pluribus/bus/events.ndjson"):
        self.agent_id = agent_id
        self.bus_path = bus_path
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    def start(
        self,
        description: str,
        parent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new task."""
        task_id = str(uuid.uuid4())
        
        self.active_tasks[task_id] = {
            "id": task_id,
            "parent_id": parent_id,
            "description": description,
            "status": "RUNNING",
            "progress": 0.0,
            "started_at": time.time(),
            "context": context or {},
        }
        
        self._emit(task_id, "RUNNING", 0.0)
        return task_id
    
    def update(
        self,
        task_id: str,
        progress: float,
        context_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update task progress."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        task["progress"] = progress
        if context_update:
            task["context"].update(context_update)
        
        self._emit(task_id, "RUNNING", progress)
    
    def complete(self, task_id: str, result: Any = None) -> None:
        """Mark task as completed."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        task["status"] = "COMPLETED"
        task["progress"] = 1.0
        task["completed_at"] = time.time()
        task["result"] = result
        
        self._emit(task_id, "COMPLETED", 1.0)
        del self.active_tasks[task_id]
    
    def fail(self, task_id: str, error: str) -> None:
        """Mark task as failed."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        task["status"] = "FAILED"
        task["error"] = error
        task["failed_at"] = time.time()
        
        self._emit(task_id, "FAILED", task["progress"], error=error)
        del self.active_tasks[task_id]
    
    def checkpoint(self, task_id: str, checkpoint_data: Dict[str, Any]) -> None:
        """Emit checkpoint for mid-task resumption."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        self._emit(
            task_id,
            "RUNNING",
            task["progress"],
            checkpoint=checkpoint_data,
        )
    
    def _emit(
        self,
        task_id: str,
        status: str,
        progress: float,
        error: Optional[str] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit task lifecycle event."""
        task = self.active_tasks.get(task_id, {})
        
        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": f"agent.{self.agent_id}.task",
            "kind": "task_lifecycle",
            "level": "info" if status != "FAILED" else "error",
            "actor": self.agent_id,
            "data": {
                "task_id": task_id,
                "parent_id": task.get("parent_id"),
                "status": status,
                "progress": progress,
                "description": task.get("description"),
                "meta": {
                    "context": task.get("context"),
                    "error": error,
                    "checkpoint": checkpoint,
                }
            }
        }
        
        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[TaskLifecycle] Failed to emit: {e}")


__all__ = [
    "PROVIDER_PRIORITY",
    "DispatchRequest",
    "SwarmDispatcher",
    "TaskLifecycle",
]
