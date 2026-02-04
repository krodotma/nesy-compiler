#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# Ensure we can import from nucleus/tools
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from graphiti_bridge import GraphitiService

def migrate():
    print("Migrating Sagent KG to Graphiti (v26)...")
    root = Path("/pluribus")
    legacy_kg = root / ".pluribus/index/kg_nodes.ndjson"
    service = GraphitiService(root)
    
    if not legacy_kg.exists():
        print("Legacy KG not found. Migration skipped.")
        return

    count = 0
    with open(legacy_kg, "r") as f:
        for line in f:
            try:
                node = json.loads(line)
                if node.get("type") == "phenotype_projection":
                    # Convert node to fact
                    # Subject: Pattern, Predicate: has_projection, Object: Projection
                    res = service.add_fact(
                        subject="pattern:sota:aicodeking",
                        predicate="has_projection",
                        object_value=node["id"],
                        source="migration:sagent",
                        confidence=1.0,
                        provenance_id=node.get("id") # Use its own ID as provenance for now
                    )
                    if "error" not in res:
                        count += 1
            except: continue
            
    print(f"Migrated {count} projections to Graphiti.")

if __name__ == "__main__":
    migrate()
