#!/usr/bin/env python3
"""
System Verification Tool (Epoch 2)
==================================

Comprehensive health check for the Pluribus Rhizome.
Verifies that all subsystems ($G_1$ - $G_{11}$) are online and operational.

Usage:
    python3 verify_system.py --emit-bus
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "nucleus" / "tools"
BUS_DIR = ROOT / ".pluribus" / "bus"

def check_tool(name: str, cmd: list, expect_in_stdout: str = "", timeout: int = 10) -> dict:
    start = time.time()
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PLURIBUS_BUS_DIR": str(BUS_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
        )
        ok = res.returncode == 0
        if ok and expect_in_stdout:
            ok = expect_in_stdout in res.stdout
        
        return {
            "status": "ok" if ok else "error",
            "latency_ms": (time.time() - start) * 1000,
            "error": res.stderr.strip() if not ok else None
        }
    except Exception as e:
        return {
            "status": "fail",
            "latency_ms": (time.time() - start) * 1000,
            "error": str(e)
        }

def verify_all(emit: bool = False):
    print("Verifying Pluribus System Fidelity...")
    
    report = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "components": {}
    }

    # 1. Lens/Collimator (G1)
    # Correct signature: req_id, goal, kind, effects, prefer_providers, require_model_prefix
    lens_script = f"""
import sys; sys.path.append('{str(TOOLS)}'); 
from lens_collimator import LensRequest, plan_route; 
print(plan_route(LensRequest(req_id='test', goal='code', kind='apply', effects='none', prefer_providers=['auto'], require_model_prefix=None), session={{}}))
"""
    report["components"]["lens"] = check_tool(
        "Lens", [sys.executable, "-c", lens_script], "depth="
    )

    # 2. PluriChat (G2/G6)
    # Web-session-only policy: status probe should not rely on mock/CLI/API providers.
    report["components"]["plurichat"] = check_tool(
        "PluriChat", [sys.executable, str(TOOLS / "plurichat.py"), "--status"], "Providers"
    )

    # 3. STRp Topology (G5)
    # Check star topology reproduction script (longer timeout)
    report["components"]["strp"] = check_tool(
        "STRp", [sys.executable, str(TOOLS / "reproduce_strp_star.py")], "Success: True", timeout=30
    )

    # 4. RAG/Memory (G3)
    # Check vector stats
    report["components"]["rag"] = check_tool(
        "RAG", [sys.executable, str(TOOLS / "rag_vector.py"), "stats"], "vec_available"
    )

    # 5. Art Dept (G11)
    # Check if manifesto exists (simple file check)
    art_manifesto = ROOT / "nucleus" / "art_dept" / "MANIFESTO.md"
    report["components"]["art_dept"] = {
        "status": "ok" if art_manifesto.exists() else "error",
        "latency_ms": 0,
        "detail": "Manifesto found" if art_manifesto.exists() else "Manifesto missing"
    }

    # 6. Teleology (G9)
    # Check domain registry count
    domains_path = ROOT / ".pluribus" / "index" / "domains.ndjson"
    count = 0
    if domains_path.exists():
        with open(domains_path, "r") as f:
            count = sum(1 for _ in f)
    report["components"]["teleology"] = {
        "status": "ok",
        "count": count,
        "detail": f"{count} domains registered"
    }

    # 7. OITERATE (G8/G10)
    report["components"]["oiterate"] = check_tool(
        "OITERATE", [sys.executable, str(TOOLS / "oiterate_operator.py"), "--help"], "usage"
    )

    print(json.dumps(report, indent=2))

    if emit:
        # Emit to bus
        event = {
            "id": str(uuid.uuid4()),
            "ts": report["ts"],
            "iso": report["iso"],
            "topic": "system.health",
            "kind": "metric",
            "level": "info",
            "actor": os.environ.get("PLURIBUS_ACTOR", "verify_system"),
            "data": report
        }
        with open(BUS_DIR / "events.ndjson", "a") as f:
            f.write(json.dumps(event) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emit-bus", action="store_true")
    args = p.parse_args()
    verify_all(args.emit_bus)

if __name__ == "__main__":
    main()
