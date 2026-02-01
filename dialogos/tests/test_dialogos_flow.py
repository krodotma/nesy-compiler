

import asyncio
import sys
import os
import time
import json
import uuid
import subprocess
import signal

# Add nucleus/tools to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../nucleus/tools')))

from agent_bus import AgentBus, BusEvent, Topics, emit_bus_event

async def test_dialogos_flow():
    print("🧪 Starting Dialogos E2E Test (Subprocess Mode)...")
    
    # 1. Start Runner as Subprocess
    runner_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '../nucleus/tools/dialogos_runner.py'))
    print(f"🚀 Launching runner: {runner_script}")
    
    runner_process = subprocess.Popen(
        [sys.executable, "-u", runner_script],
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid  # Allow killing process group
    )
    
    try:
        # Allow runner to spin up
        print("⏳ Waiting for runner initialization...")
        await asyncio.sleep(2)
        
        # 2. Emit Request
        bus = AgentBus()
        req_id = str(uuid.uuid4())
        print(f"📤 Emitting request: {req_id}")
        
        emit_bus_event(
            topic=f"{Topics.DIALOGOS_SUBMIT}.claude",
            actor="test_harness",
            data={
                "req_id": req_id,
                "provider": "claude-opus", 
                "prompt": "Hello world check",
            }
        )
        
        # 3. Listen for Response
        print("👂 Listening for response...")
        response_received = False
        
        start_wait = time.time()
        # Create a new iterator instead of sharing potentially stale state
        for event in bus.watch(Topics.DIALOGOS_CELL):
            if event.topic == f"{Topics.DIALOGOS_CELL}.claude-opus" and event.data.get("req_id") == req_id:
                print(f"✅ Received response for {req_id}")
                print(f"   Payload: {event.data.get('response')}")
                response_received = True
                break
                
            if time.time() - start_wait > 10:
                print("❌ Timeout waiting for response")
                break
    
    finally:
        # 4. cleanup
        print("🛑 Terminating runner...")
        os.killpg(os.getpgid(runner_process.pid), signal.SIGTERM)
        runner_process.wait()
    
    if response_received:
        print("🎉 Test Passed")
        sys.exit(0)
    else:
        print("💥 Test Failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_dialogos_flow())
