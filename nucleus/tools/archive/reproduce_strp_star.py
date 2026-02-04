#!/usr/bin/env python3
"""
Reproduction script for STRp Star Topology (Iso-STRp)
"""
import sys
import json
import uuid
import os
import subprocess
from pathlib import Path

# Add tools dir to path
sys.path.append(str(Path(__file__).parent))

from strp_topology import StarTopologyExecutor, STRpRequest

def main():
    print("Starting STRp Star Topology Reproduction...")
    
    # Mock bus setup
    bus_dir = Path("/pluribus/.pluribus/bus")
    os.environ["PLURIBUS_BUS_DIR"] = str(bus_dir)
    os.environ["PLURIBUS_ACTOR"] = "repro-script"
    
    # Define a complex task requiring decomposition
    task = "Compare the architectural differences between Kubernetes, Nomad, and Docker Swarm for a stateful database workload."
    req_id = f"repro-star-{uuid.uuid4().hex[:8]}"
    
    print(f"Request ID: {req_id}")
    print(f"Task: {task}")
    
    request = STRpRequest(
        req_id=req_id,
        task=task,
        topology="star",
        fanout=3,
        timeout_s=30.0,
        isolation="process", # Force process isolation to test IsoExecutor path if used
        coordinator="mock",  # Use mock to avoid external API calls during repro
        workers=["mock", "mock", "mock"]
    )
    
    executor = StarTopologyExecutor()
    
    print("\nExecuting Star Topology...")
    response = executor.execute(request)
    
    print("\n--- Response ---")
    print(f"Success: {response.success}")
    print(f"Provider: {response.provider}")
    print(f"Content Length: {len(response.content)}")
    print(f"Results Count: {len(response.results)}")
    
    print("\n--- Subtask Results ---")
    for res in response.results:
        print(f"Worker: {res.get('worker')}")
        print(f"Subtask: {res.get('subtask')[:50]}...")
        print(f"Success: {res.get('success')}")
        print("-" * 20)

    # Verify aggregation happened
    if response.success and len(response.results) == 3:
        print("\n[PASS] Star topology decomposition and aggregation successful.")
        sys.exit(0)
    else:
        print("\n[FAIL] Star topology execution incomplete.")
        sys.exit(1)

if __name__ == "__main__":
    main()
