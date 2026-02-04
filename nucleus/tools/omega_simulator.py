#!/usr/bin/env python3
"""
Geometric Omega Simulator
=========================
Simulates the "Geometric Mean God Machine Learning Omega" consensus mechanism
by emitting synthetic `omega.vector.stream` events to the agent bus.

Role:
- Demonstrates the "Pushdown Stack" visualization in `strp_monitor.py`.
- Simulates multi-agent vector convergence.
- Validates the TUI's ability to render stack depth and geometric means.

Usage:
    python3 omega_simulator.py [--bus-dir /path/to/bus]
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

# Simulation Scenarios
SCENARIOS = [
    {
        "name": "Implement PQC Verification",
        "stack": [
            "root:orchestrate",
            "secure:pqc_flow",
            "crypto:verify_signature",
            "fs:read_file"
        ],
        "vectors": [0.95, 0.98, 0.92], # High consensus
    },
    {
        "name": "Refactor Legacy Code",
        "stack": [
            "root:orchestrate",
            "refactor:analyze",
            "ast:parse",
            "error:syntax_check"
        ],
        "vectors": [0.80, 0.40, 0.85], # Divergence (one agent unsure)
    },
    {
        "name": "Web Scraping (Headless)",
        "stack": [
            "root:orchestrate",
            "web:browse",
            "bridge:connect_chrome",
            "dom:extract_text"
        ],
        "vectors": [0.99, 0.99, 0.99], # Unanimous
    }
]

def emit_event(bus_dir: str, data: dict):
    tool = Path(__file__).parent / "agent_bus.py"
    if not tool.exists():
        return
    
    subprocess.run(
        [
            sys.executable, str(tool),
            "pub",
            "--bus-dir", bus_dir,
            "--topic", "omega.vector.stream",
            "--kind", "metric",
            "--level", "info",
            "--actor", "omega_sim",
            "--data", json.dumps(data)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def geometric_mean(vectors):
    if not vectors:
        return 0.0
    product = 1.0
    for v in vectors:
        product *= v
    return product ** (1.0 / len(vectors))

def run_simulation(bus_dir: str):
    print(f"Starting Omega Simulator... (Bus: {bus_dir})")
    print("Press Ctrl+C to stop.")
    
    while True:
        scenario = random.choice(SCENARIOS)
        
        # Simulate stack push/pop dynamics
        current_stack = []
        for depth, item in enumerate(scenario["stack"]):
            current_stack.insert(0, {"item": item, "depth": depth, "state": "active"})
            
            # Base vectors + random noise (Aleatoric uncertainty)
            vectors = [min(1.0, max(0.0, v + random.uniform(-0.05, 0.05))) for v in scenario["vectors"]]
            gm = geometric_mean(vectors)
            
            payload = {
                "scenario": scenario["name"],
                "stack": current_stack,
                "stack_top": item,
                "stack_depth": depth,
                "vectors": vectors,
                "geometric_mean": gm,
                "tensor_shard": f"sim:{time.time()}"
            }
            
            emit_event(bus_dir, payload)
            time.sleep(0.8) # Human-readable speed
            
        time.sleep(2.0) # Pause between tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bus-dir", default=os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    args = parser.parse_args()
    
    try:
        run_simulation(args.bus_dir)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
