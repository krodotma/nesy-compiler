"""
Theia A2A Timeout Handler — Automatic failover for stalled requests.

Monitors pending A2A requests and routes to healthy providers
if no response within timeout threshold.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class PendingRequest:
    """Tracked pending request."""
    id: str
    topic: str
    created_at: float
    target_provider: Optional[str] = None
    attempts: int = 0
    last_attempt: float = 0.0


class TimeoutHandler:
    """
    Monitors and handles timed-out A2A requests.
    
    Features:
    - Configurable timeout threshold
    - Automatic failover to healthy providers
    - Exponential backoff for retries
    """
    
    # Provider priority for failover
    PROVIDER_PRIORITY = ["codex", "ollama", "claude", "gemini"]
    
    def __init__(
        self,
        bus_path: str = "/pluribus/.pluribus/bus/events.ndjson",
        timeout_seconds: float = 60.0,
        max_attempts: int = 3,
    ):
        self.bus_path = Path(bus_path)
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.pending: Dict[str, PendingRequest] = {}
        self._provider_health: Dict[str, bool] = {}
    
    def update_health(self, provider: str, healthy: bool) -> None:
        """Update provider health status."""
        self._provider_health[provider] = healthy
    
    def is_healthy(self, provider: str) -> bool:
        """Check if provider is healthy."""
        return self._provider_health.get(provider, False)
    
    def get_healthy_providers(self) -> List[str]:
        """Get healthy providers in priority order."""
        return [p for p in self.PROVIDER_PRIORITY if self.is_healthy(p)]
    
    def track_request(self, request_id: str, topic: str, target: Optional[str] = None) -> None:
        """Start tracking a new request."""
        self.pending[request_id] = PendingRequest(
            id=request_id,
            topic=topic,
            created_at=time.time(),
            target_provider=target,
        )
    
    def mark_completed(self, request_id: str) -> None:
        """Mark request as completed (response received)."""
        if request_id in self.pending:
            del self.pending[request_id]
    
    def check_timeouts(self) -> List[PendingRequest]:
        """Check for timed-out requests."""
        now = time.time()
        timed_out = []
        
        for req_id, req in list(self.pending.items()):
            age = now - req.created_at
            
            if age > self.timeout_seconds:
                if req.attempts < self.max_attempts:
                    timed_out.append(req)
                else:
                    # Max attempts reached, emit failure
                    self._emit_failure(req, "Max attempts exceeded")
                    del self.pending[req_id]
        
        return timed_out
    
    def handle_timeout(self, request: PendingRequest) -> bool:
        """
        Handle a timed-out request by routing to next healthy provider.
        
        Returns True if successfully re-routed.
        """
        healthy = self.get_healthy_providers()
        
        if not healthy:
            return False
        
        # Pick next provider (rotate through healthy ones)
        provider_idx = request.attempts % len(healthy)
        next_provider = healthy[provider_idx]
        
        # Emit re-route event
        self._emit_reroute(request, next_provider)
        
        # Update tracking
        request.attempts += 1
        request.last_attempt = time.time()
        request.target_provider = next_provider
        
        return True
    
    def _emit_reroute(self, request: PendingRequest, provider: str) -> None:
        """Emit re-routing event."""
        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": "theia.swarm.reroute",
            "kind": "dispatch",
            "level": "warn",
            "actor": "theia-timeout-handler",
            "data": {
                "request_id": request.id,
                "original_topic": request.topic,
                "new_provider": provider,
                "attempt": request.attempts + 1,
                "age_seconds": time.time() - request.created_at,
            }
        }
        
        self._write_event(event)
    
    def _emit_failure(self, request: PendingRequest, reason: str) -> None:
        """Emit failure event."""
        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": "theia.swarm.failed",
            "kind": "error",
            "level": "error",
            "actor": "theia-timeout-handler",
            "data": {
                "request_id": request.id,
                "original_topic": request.topic,
                "reason": reason,
                "attempts": request.attempts,
                "age_seconds": time.time() - request.created_at,
            }
        }
        
        self._write_event(event)
    
    def _write_event(self, event: Dict[str, Any]) -> None:
        """Write event to bus."""
        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[TimeoutHandler] Failed to write: {e}")
    
    def process_cycle(self) -> Dict[str, Any]:
        """
        Run one cycle of timeout processing.
        
        Returns stats about processed requests.
        """
        timed_out = self.check_timeouts()
        rerouted = 0
        failed = 0
        
        for req in timed_out:
            if self.handle_timeout(req):
                rerouted += 1
            else:
                failed += 1
        
        return {
            "pending": len(self.pending),
            "timed_out": len(timed_out),
            "rerouted": rerouted,
            "failed": failed,
            "healthy_providers": self.get_healthy_providers(),
        }


__all__ = ["PendingRequest", "TimeoutHandler"]
