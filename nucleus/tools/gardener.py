#!/usr/bin/env python3
"""
Gardener Operator: The Soul of the Cell
=======================================

The Gardener is the autonomous agent that "wakes up" to tend the Rhizome.
It scans for domains in the 'Concept' phase (Phase 1) and actualizes them
into 'Structure' (Phase 2) by dreaming their initial artifacts.

Usage:
    python3 gardener.py tend --limit 3
"""

import argparse
import json
import os
import sys
import time
import uuid
import random
from pathlib import Path

sys.dont_write_bytecode = True

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def get_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "gardener"

def load_registry(root: Path) -> list[dict]:
    reg_path = root / ".pluribus" / "index" / "domains.ndjson"
    if not reg_path.exists():
        return []
    domains = []
    with reg_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                domains.append(json.loads(line))
            except:
                continue
    return domains

def emit_bus(bus_dir: Path, topic: str, data: dict, actor: str) -> None:
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": "artifact",
        "level": "info",
        "actor": actor,
        "data": data,
    }
    events_path = bus_dir / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def dream_structure(domain: str, telos: str, tags: list) -> str:
    """Simulate the creative act of structural generation."""
    # In a real system, this would call an LLM (Codex/Claude).
    # Here we structurally hallucinate a valid scaffold.
    return f"""# {domain}

> {telos}

## Overview
Auto-generated structure for **{domain}**.
Tags: {', '.join(tags)}

## Architecture
1. **Core**: Teleological alignment.
2. **Interface**: Isomorphic access.
3. **State**: Bus-driven evidence.

*Dreamt by Gardener at {now_iso()}*"""

def cmd_tend(args):
    root = Path(args.root or "/pluribus")
    bus_dir = Path(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR", str(root / ".pluribus" / "bus")))
    
    domains = load_registry(root)
    candidates = [d for d in domains if d.get("teleology", {}).get("phase") == "concept"]
    
    if not candidates:
        print("The garden is tended. No concepts waiting.")
        return

    # Shuffle for aleatoric distribution
    random.shuffle(candidates)
    selected = candidates[:int(args.limit)]
    
    print(f"Tending to {len(selected)} concepts...")
    
    for d in selected:
        dom = d["domain"]
        teleology = d["teleology"]
        
        print(f"  Cultivating {dom}...")
        
        # 1. Dream
        readme_content = dream_structure(dom, teleology["telos"], d.get("tags", []))
        
        # 2. Actualize (File creation)
        # We map domains to a 'rhizome/domains' folder for now
        domain_dir = root / "rhizome" / "domains" / dom
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / "README.md").write_text(readme_content, encoding="utf-8")
        
        # 3. Evolve Phase
        # We emit the intent to update the registry (idempotent ledger)
        # In a full system, we'd write back to the registry, but here we emit the artifact event
        emit_bus(bus_dir, "gardener.cultivated", {
            "domain": dom,
            "previous_phase": "concept",
            "new_phase": "structure",
            "artifact": str(domain_dir / "README.md")
        }, get_actor())
        
        print(f"    -> Structure created at {domain_dir / 'README.md'}")

def main():
    parser = argparse.ArgumentParser(description="Gardener Operator")
    parser.add_argument("--root", help="Project root")
    parser.add_argument("--bus-dir", help="Bus directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    tend_p = subparsers.add_parser("tend", help="Tend to concepts")
    tend_p.add_argument("--limit", default=3, help="Max domains to process")
    
    args = parser.parse_args()
    
    if args.command == "tend":
        cmd_tend(args)

if __name__ == "__main__":
    main()
