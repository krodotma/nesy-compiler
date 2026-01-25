#!/usr/bin/env python3
"""
laser_direct.py - LASER Direct Runner

Language Augmented Superpositional Effective Retrieval (LASER).
Primary engine for phenomenological verification and autonomous feedback.
Bypasses legacy shims for direct provider access.

Usage:
  python3 laser_direct.py --prompt "Verify button color" --model "claude-opus"
  python3 laser_direct.py --watch  # Listen for browser.chat.request events
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

# Local imports
try:
    from agent_bus import AgentBus, BusEvent, EventKind, EventLevel, emit_bus_event
    from gymnist import Gymnist, GymRequest, GymResponse
except ImportError:
    # Bootstrap path for dev execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nucleus.tools.agent_bus import AgentBus, BusEvent, EventKind, EventLevel, emit_bus_event
    from nucleus.tools.gymnist import Gymnist, GymRequest, GymResponse


@dataclass
class LaserResult:
    """Result of a LASER operation."""
    req_id: str
    provider: str
    content: str
    success: bool
    latency_ms: int
    metadata: dict = field(default_factory=dict)


class LaserDirect:
    """
    LASER Direct runner.
    Integrates with Gymnist for provider abstraction (Gymnist handles the keys/providers).
    Adds the layer of "Superpositional Verification" - running tasks and verify results.
    """
    
    def __init__(self, mock_mode: bool = False):
        self.bus = AgentBus()
        self.gym = Gymnist()
        self.mock_mode = mock_mode
        self.running = False

    async def execute_prompt(
        self,
        prompt: str,
        model: str = "claude-opus-4",
        req_id: Optional[str] = None
    ) -> LaserResult:
        """Execute a direct prompt."""
        req_id = req_id or str(uuid.uuid4())
        start_time = time.time()
        
        print(f"🔦 LASER: {model} <-- {prompt[:50]}...")
        
        try:
            if self.mock_mode:
                # Mock response
                await asyncio.sleep(0.5)
                content = f"[LASER MOCK] Processed: {prompt[:30]}..."
                latency = int((time.time() - start_time) * 1000)
                return LaserResult(
                    req_id=req_id,
                    provider=model,
                    content=content,
                    success=True,
                    latency_ms=latency,
                    metadata={"mock": True}
                )

            # Use Gymnist for real/stubbed call
            # Note: Gymnist will use OPENAI_API_KEY if model is mapped to OpenAI,
            # or gracefully stub others if keys missing.
            response = await self.gym.complete(model, prompt)
            
            return LaserResult(
                req_id=req_id,
                provider=model,
                content=response.content,
                success=True,
                latency_ms=response.latency_ms,
                metadata=response.usage
            )
            
        except Exception as e:
            latency = int((time.time() - start_time) * 1000)
            return LaserResult(
                req_id=req_id,
                provider=model,
                content=str(e),
                success=False,
                latency_ms=latency,
                metadata={"error": str(e)}
            )

    async def process_bus_request(self, event: BusEvent):
        """Handle browser.chat.request events."""
        data = event.data
        prompt = data.get("prompt")
        req_id = data.get("req_id", str(uuid.uuid4()))
        model = data.get("model", "claude-opus-4")
        
        if not prompt:
            return

        result = await self.execute_prompt(prompt, model, req_id)
        
        # Emit response
        emit_bus_event(
            topic="browser.chat.response",
            actor="laser_direct",
            data={
                "req_id": result.req_id,
                "content": result.content,
                "success": result.success,
                "latency_ms": result.latency_ms,
                "provider": result.provider
            },
            kind=EventKind.RESPONSE,
            level=EventLevel.INFO if result.success else EventLevel.ERROR
        )

    async def watch(self):
        """Watch bus for requests."""
        self.running = True
        print(f"🔭 LASER Direct watching bus...")
        print(f"   Topic: browser.chat.request")
        
        try:
            for event in self.bus.watch("browser.chat.request"):
                if not self.running:
                    break
                # Process async to not block bus reader? 
                # For simplicity in this v1, await directly
                await self.process_bus_request(event)
        except KeyboardInterrupt:
            print("\n🛑 LASER Direct stopping...")

    def stop(self):
        self.running = False


async def main():
    parser = argparse.ArgumentParser(description="LASER Direct Runner")
    parser.add_argument("--prompt", help="Direct prompt update")
    parser.add_argument("--model", default="claude-opus-4", help="Target model")
    parser.add_argument("--watch", action="store_true", help="Watch bus for requests")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no API calls)")
    
    args = parser.parse_args()
    
    laser = LaserDirect(mock_mode=args.mock)
    
    if args.watch:
        await laser.watch()
    elif args.prompt:
        result = await laser.execute_prompt(args.prompt, args.model)
        status = "✅" if result.success else "❌"
        print(f"\n{status} Result ({result.latency_ms}ms):\n{result.content}")
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
