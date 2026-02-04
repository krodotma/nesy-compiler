#!/usr/bin/env python3
"""
Teleology Tool: The Purpose of Names
====================================

Manages the lifecycle of domains from 'Concept' to 'Organism'.
Injects teleological metadata and generates aleatoric prompts for gap filling.

Usage:
    python3 teleology.py scan --auto-classify
    python3 teleology.py inject --domain boddah.org --phase structure --telos "Refutation of Last Days"
    python3 teleology.py aleatoric --domain boddah.org --persona "ring0.architect"
"""

import argparse
import json
import os
import sys
import time
import uuid
import random
from pathlib import Path
from typing import Optional

sys.dont_write_bytecode = True

# Constants
PHASES = ['concept', 'structure', 'actualizing', 'mapped', 'evolving']

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def get_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "teleology-tool"

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

def save_registry(root: Path, domains: list[dict]) -> None:
    reg_path = root / ".pluribus" / "index" / "domains.ndjson"
    with reg_path.open("w", encoding="utf-8") as f:
        for d in domains:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

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

def cmd_scan(args):
    root = Path(args.root or "/pluribus")
    bus_dir = Path(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR", str(root / ".pluribus" / "bus")))
    
    domains = load_registry(root)
    updated_count = 0
    
    print(f"Scanning {len(domains)} domains for teleology...")
    
    for d in domains:
        if "teleology" not in d and args.auto_classify:
            # Heuristic classification
            tags = d.get("tags", [])
            phase = "concept"
            
            if "sota" in tags or "devstack" in tags:
                phase = "structure"
            
            # Infer Telos from tags
            telos = f"Realize {', '.join(tags)}"
            
            d["teleology"] = {
                "phase": phase,
                "telos": telos,
                "virtues": {k: False for k in ["semantic", "semiotic", "engineering", "architectural", "constitutional", "entropic"]},
                "provenance": {"origin": "human", "intent_vector": "inferred"},
                "gaps": ["structure definition", "implementation"],
                "aleatoric_seed": str(uuid.uuid4())[:8]
            }
            updated_count += 1
            print(f"  Injected teleology for {d['domain']} ({phase})")
            
            emit_bus(bus_dir, "teleology.injected", {
                "domain": d["domain"],
                "teleology": d["teleology"]
            }, get_actor())

    if updated_count > 0:
        save_registry(root, domains)
        print(f"Updated {updated_count} domains.")
    else:
        print("No updates needed.")

def cmd_aleatoric(args):
    root = Path(args.root or "/pluribus")
    domains = load_registry(root)
    
    target = next((d for d in domains if d.get("domain") == args.domain), None)
    if not target:
        print(f"Domain {args.domain} not found.")
        return
        
    teleology = target.get("teleology", {})
    if not teleology:
        print(f"No teleology for {args.domain}. Run scan first.")
        return
        
    seed = teleology.get("aleatoric_seed", "default")
    random.seed(seed)
    
    # Generate an "Epistemic Gap" prompt
    persona = args.persona or "ring0.architect"
    tags_str = ", ".join(target.get("tags", []))
    
    prompt = f"""
[Aleatoric Prompt for {args.domain}]
Phase: {teleology.get('phase')}
Telos: {teleology.get('telos')}
Seed: {seed}

Identify the critical epistemic gaps preventing this domain from evolving to the next phase.
Propose a structural hypothesis or ontology that bridges 'Word' to 'Organism'.
Focus on: {tags_str}

Constraints:
- Must be semiotically consistent.
- Must be architecturally compliant (Ring/Sextet).
- Must define anti-scaling factors (entropy limits).
"""
    print(prompt)
    
    # In a real system, we'd dispatch this to the bus.
    # For now, we emit the intent.
    bus_dir = Path(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR", str(root / ".pluribus" / "bus")))
    emit_bus(bus_dir, "teleology.aleatoric.prompt", {
        "domain": args.domain,
        "prompt": prompt,
        "persona": persona
    }, get_actor())

def main():
    parser = argparse.ArgumentParser(description="Teleology Tool")
    parser.add_argument("--root", help="Project root")
    parser.add_argument("--bus-dir", help="Bus directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    scan_p = subparsers.add_parser("scan", help="Scan and inject teleology")
    scan_p.add_argument("--auto-classify", action="store_true", help="Auto-infer phase/telos")
    
    aleatoric_p = subparsers.add_parser("aleatoric", help="Generate aleatoric prompt")
    aleatoric_p.add_argument("--domain", required=True)
    aleatoric_p.add_argument("--persona", help="Target persona")
    
    args = parser.parse_args()
    
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "aleatoric":
        cmd_aleatoric(args)

if __name__ == "__main__":
    main()