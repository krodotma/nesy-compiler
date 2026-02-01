
#!/usr/bin/env python3
"""
dialogos_runner.py - Core Message Dispatch System

Dialogos owns all agent-to-LLM communication. This runner:
1. Watches the bus for dialogos.submit.* requests
2. Routes to appropriate provider via gymnist
3. Emits dialogos.cell.* responses
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# Local imports
from agent_bus import AgentBus, BusEvent, EventKind, EventLevel, Topics, emit_bus_event
from providers import ProviderFactory

# RAK: Ralph exit detection patterns
import re

def check_fix_plan_complete(fix_plan_path: str = "@fix_plan.md") -> bool:
    """
    Ralph-style exit detection: check if all checkboxes in @fix_plan.md are complete.
    
    Returns True if file exists and all [ ] are now [x].
    """
    path = Path(fix_plan_path)
    if not path.exists():
        return False
    
    content = path.read_text()
    unchecked = re.findall(r"- \[ \]", content)
    checked = re.findall(r"- \[x\]", content, re.IGNORECASE)
    
    # Complete if no unchecked and at least one checked
    return len(unchecked) == 0 and len(checked) > 0


def count_done_signals(response: str) -> int:
    """Count completion signals in response text."""
    signals = [
        r"<promise>DONE</promise>",
        r"task\s+complete",
        r"all\s+tasks?\s+done",
        r"finished\s+all",
    ]
    count = 0
    for pattern in signals:
        if re.search(pattern, response, re.IGNORECASE):
            count += 1
    return count


class RalphExitTracker:
    """Track consecutive done signals for Ralph-style exit detection."""
    
    def __init__(self, threshold: int = 3):
        self.consecutive_count = 0
        self.threshold = threshold
    
    def check(self, response: str) -> bool:
        """Returns True if exit conditions met."""
        signals = count_done_signals(response)
        if signals > 0:
            self.consecutive_count += signals
        else:
            self.consecutive_count = 0  # Reset on non-done response
        
        return self.consecutive_count >= self.threshold
    
    def reset(self):
        self.consecutive_count = 0


class CircuitBreaker:
    """
    Agent C: Robustness - Circuit Breaker.
    Prevents cascade failures by opening circuit after N consecutive failures.
    """
    def __init__(self, failure_threshold: int = 3, reset_timeout_s: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED" # CLOSED, OPEN, HALF_OPEN

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            print(f"🔌 Circuit OPENED (Failures: {self.failures})")

    def record_success(self):
        if self.state != "CLOSED":
            print("🔌 Circuit RESET to CLOSED")
        self.failures = 0
        self.state = "CLOSED"

    def allow_request(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout_s:
                self.state = "HALF_OPEN"
                return True # Allow one trial
            return False
        return True # HALF_OPEN (trial)


class ProviderStatus(Enum):
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class DialogosRequest:
    """Parsed dialogos request from bus."""
    req_id: str
    provider: str
    prompt: str
    context: dict = field(default_factory=dict)
    timeout_s: int = 120
    actor: str = "unknown"
    
    @classmethod
    def from_bus_event(cls, event: BusEvent) -> "DialogosRequest":
        data = event.data
        return cls(
            req_id=data.get("req_id", str(uuid.uuid4())),
            provider=data.get("provider", "claude-opus"),
            prompt=data.get("prompt", ""),
            context=data.get("context", {}),
            timeout_s=data.get("timeout_s", 120),
            actor=event.actor,
        )


@dataclass  
class DialogosResponse:
    """Response structure for dialogos cell."""
    req_id: str
    provider: str
    success: bool
    response: str = ""
    error: str = ""
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0


class ProviderRegistry:
    """Registry of available LLM providers via Factory."""
    
    # Metadata for routing/scheduling
    PROVIDERS_META = {
        "claude-opus": {
            "name": "Claude Opus",
            "model": "claude-3-opus-20240229",
            "rate_limit_rpm": 10,
            "priority": 1,
        },
        "claude-sonnet": {
            "name": "Claude Sonnet", 
            "model": "claude-3-sonnet-20240229",
            "rate_limit_rpm": 50,
            "priority": 2,
        },
        "gemini-2": {
            "name": "Gemini 3 Pro",
            "model": "gemini-3-pro",
            "provider": "google",
        },
        "kimi-2-5": {
            "name": "Kimi 2.5",
            "model": "kimi-2.5",
            "rate_limit_rpm": 60,
            "priority": 3,
        },
        "qwen-plus": {
            "name": "Qwen Plus",
            "model": "qwen-plus",
            "rate_limit_rpm": 100,
            "priority": 3,
        },
    }
    
    def __init__(self):
        self.status: dict[str, ProviderStatus] = {
            p: ProviderStatus.READY for p in self.PROVIDERS_META
        }
        self.last_request: dict[str, float] = {}
        # One breaker per provider
        self.breakers: dict[str, CircuitBreaker] = {
            p: CircuitBreaker() for p in self.PROVIDERS_META
        }

    
    def is_available(self, provider_id: str) -> bool:
        breaker_ok = self.breakers.get(provider_id, CircuitBreaker()).allow_request()
        return breaker_ok and self.status.get(provider_id) == ProviderStatus.READY

    
    def mark_busy(self, provider_id: str):
        self.status[provider_id] = ProviderStatus.BUSY
        self.last_request[provider_id] = time.time()
    
    def mark_ready(self, provider_id: str):
        self.status[provider_id] = ProviderStatus.READY
    
    def mark_error(self, provider_id: str):
        self.status[provider_id] = ProviderStatus.ERROR
        if provider_id in self.breakers:
            self.breakers[provider_id].record_failure()

    def mark_success(self, provider_id: str):
        if provider_id in self.breakers:
            self.breakers[provider_id].record_success()



class DialogosRunner:
    """Main Dialogos message dispatch runner."""
    
    def __init__(self, bus: Optional[AgentBus] = None):
        self.bus = bus or AgentBus()
        self.registry = ProviderRegistry()
        self.pending_requests: dict[str, DialogosRequest] = {}
        self.running = False
        
    async def process_request(self, request: DialogosRequest) -> DialogosResponse:
        """Process a dialogos request through the appropriate provider."""
        start_time = time.time()
        
        # Check provider availability
        if not self.registry.is_available(request.provider):
            return DialogosResponse(
                req_id=request.req_id,
                provider=request.provider,
                success=False,
                error=f"Provider {request.provider} not available",
            )
        
        # Mark provider busy
        self.registry.mark_busy(request.provider)
        
        try:
            # Get actual provider adapter
            provider = ProviderFactory.get_provider(request.provider)
            
            if not provider:
                raise ValueError(f"Unknown provider ID: {request.provider}")
                
            # EXECUTE GENERATION
            response_text = await provider.generate(request.prompt, request.context)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return DialogosResponse(
                req_id=request.req_id,
                provider=request.provider,
                success=True,
                response=response_text,
                latency_ms=latency_ms,
                # Naive token count for now
                usage={"prompt_tokens": len(request.prompt) // 4, "completion_tokens": len(response_text) // 4},
            )
            
            # Agent C: Record success
            self.registry.mark_success(request.provider)
            return response

            
        except Exception as e:
            self.registry.mark_error(request.provider)
            return DialogosResponse(
                req_id=request.req_id,
                provider=request.provider,
                success=False,
                error=str(e),
            )
        finally:
            self.registry.mark_ready(request.provider)
    
    def emit_response(self, response: DialogosResponse):
        """Emit response to bus."""
        topic = f"{Topics.DIALOGOS_CELL}.{response.provider}"
        emit_bus_event(
            topic=topic,
            actor="dialogos",
            data={
                "req_id": response.req_id,
                "provider": response.provider,
                "success": response.success,
                "response": response.response if response.success else None,
                "error": response.error if not response.success else None,
                "usage": response.usage,
                "latency_ms": response.latency_ms,
            },
            kind=EventKind.RESPONSE,
            level=EventLevel.INFO if response.success else EventLevel.ERROR,
        )
    
    async def handle_event(self, event: BusEvent):
        """Handle incoming bus event."""
        if not event.topic.startswith(Topics.DIALOGOS_SUBMIT):
            return
        
        request = DialogosRequest.from_bus_event(event)
        self.pending_requests[request.req_id] = request
        
        print(f"📨 Received request {request.req_id[:8]} from {request.actor} → {request.provider}")
        
        response = await self.process_request(request)
        self.emit_response(response)
        
        status = "✅" if response.success else "❌"
        print(f"{status} Completed {request.req_id[:8]} in {response.latency_ms}ms")
        
        if request.req_id in self.pending_requests:
            del self.pending_requests[request.req_id]
    
    async def run(self):
        """Main event loop - watch bus and process requests."""
        self.running = True
        print("🎯 Dialogos Runner started")
        print(f"📡 Watching bus at: {self.bus.events_file}")
        
        # Announce providers
        providers_list = list(self.registry.PROVIDERS_META.keys())
        print(f"🔌 Providers: {providers_list}")
        
        emit_bus_event(
            topic="dialogos.runner.started",
            actor="dialogos",
            data={"providers": providers_list},
        )
        
        try:
            for event in self.bus.watch(Topics.DIALOGOS_SUBMIT):
                if not self.running:
                    break
                await self.handle_event(event)
        except KeyboardInterrupt:
            print("\n🛑 Dialogos Runner stopping...")
        finally:
            self.running = False
            emit_bus_event(
                topic="dialogos.runner.stopped",
                actor="dialogos",
                data={},
            )
    
    def stop(self):
        """Stop the runner."""
        self.running = False


async def main():
    """Entry point for dialogos runner."""
    runner = DialogosRunner()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
