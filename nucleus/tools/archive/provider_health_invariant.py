#!/usr/bin/env python3
"""
Provider Health Invariant (v26)
===============================

System invariant that periodically executes provider smoke tests and 
emits aggregate health metrics to the Pluribus bus.

Converts 14+ manual smoke tests into an automated liveness mesh.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
SMOKE_DIR = TOOLS_DIR / "providers"

SMOKE_TESTS = [
    "claude_cli_smoke.py",
    "codex_cli_smoke.py",
    "gemini_cli_smoke.py",
    "vertex_gemini_smoke.py",
    "ollama_smoke.py",
    "vllm_smoke.py"
]

def run_invariant_cycle():
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")
    print(f"[health] Starting provider health cycle...")
    
    results = {}
    for test in SMOKE_TESTS:
        test_path = SMOKE_DIR / test
        if not test_path.exists():
            continue
            
        try:
            res = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            status = "healthy" if res.returncode == 0 else "degraded"
            results[test.replace("_smoke.py", "")] = {
                "status": status,
                "exit_code": res.returncode,
                "latency_s": 0.0 # Future: parse from test output
            }
        except Exception as e:
            results[test.replace("_smoke.py", "")] = {"status": "error", "msg": str(e)}

    # Emit to bus
    # Note: Using agent_bus.py if available
    bus_tool = TOOLS_DIR / "agent_bus.py"
    if bus_tool.exists():
        subprocess.run([
            sys.executable, str(bus_tool), "pub",
            "--topic", "omega.providers.health",
            "--kind", "metric",
            "--actor", "health-invariant",
            "--data", json.dumps(results)
        ])

if __name__ == "__main__":
    while True:
        run_invariant_cycle()
        time.sleep(300) # Every 5 minutes
