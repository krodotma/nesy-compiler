#!/usr/bin/env python3
"""
provenance_validator.py - Verifies the Epistemic Chain of Custody (v26)

Goal: Ensure that facts in the Knowledge Graph and Vector Store have 
verifiable links back to the Bus (System of Record).

Mandate: DKIN Protocol v26 (Epistemic Sovereignty)
"""
import sys
import json
from pathlib import Path

def check_kg_provenance(kg_path: Path):
    """Check if KG facts have provenance_id."""
    if not kg_path.exists():
        return 0, 0
    
    total = 0
    linked = 0
    with open(kg_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("kind") == "graphiti_fact":
                    total += 1
                    if obj.get("provenance_id"):
                        linked += 1
            except: continue
    return total, linked

def main():
    root = Path("/pluribus")
    facts_path = root / ".pluribus" / "kg" / "graphiti_facts.ndjson"
    
    print("--- v26 Epistemic Sovereignty Audit ---")
    
    # 1. KG Check
    total, linked = check_kg_provenance(facts_path)
    print(f"KG Facts: {total}")
    print(f"Linked:   {linked} ({(linked/total*100) if total > 0 else 0:.1f}%)")
    
    if total > 0 and linked < total:
        print("\n[WARNING] Some facts lack provenance. Rebuild recommended.")
    elif total == 0:
        print("\n[INFO] KG is empty. Ready for fresh ingestion.")
    else:
        print("\n[SUCCESS] All symbolic facts are verifiable.")

if __name__ == "__main__":
    main()
