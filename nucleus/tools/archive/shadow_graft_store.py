#!/usr/bin/env python3
"""
Shadow Graft Store - Step 26 of PORTAL Implementation.
Manages the Hysteresis Buffer (SM) for lagging potentials.
"""
import sys
import json
from pathlib import Path

SHADOW_LEDGER = Path(".pluribus/index/shadow_grafts.ndjson")

def store_shadow(decoded_fragment):
    SHADOW_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with SHADOW_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decoded_fragment) + "\n")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = json.loads(sys.argv[1])
        store_shadow(data)
        print("Shadow Graft Stored.")
