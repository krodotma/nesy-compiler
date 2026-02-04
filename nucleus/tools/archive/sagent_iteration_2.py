#!/usr/bin/env python3
import json
import sys
import uuid
import time
from pathlib import Path

# Ensure we can import from nucleus/tools
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))

from nucleus.tools.event_semantics import create_semantic_event, EntelexisState, SemioticalysisLayer

# Configuration
KG_PATH = Path(".pluribus/index/kg_nodes.ndjson")
BUS_PATH = Path(".pluribus/bus/events.ndjson")

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_kg_nodes(path: Path) -> list[dict]:
    nodes = []
    if not path.exists():
        return nodes
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    nodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return nodes

def run():
    print("Starting Sagent Iteration 2...")

    # 1. Load KG
    kg_items = load_kg_nodes(KG_PATH)
    print(f"Loaded {len(kg_items)} items from KG.")

    # 2. Identify Workflows needing Phenotype Projection
    workflows = [item for item in kg_items if item.get("type") == "workflow"]
    
    # Simple check to see if projection already exists (by edge or node convention)
    # We'll use a set of source IDs that already have a 'projected_as' edge
    projected_workflow_ids = set()
    for item in kg_items:
        if item.get("rel") == "projected_as":
            projected_workflow_ids.add(item.get("from"))

    target_workflows = [wf for wf in workflows if wf["id"] not in projected_workflow_ids]
    
    print(f"Found {len(target_workflows)} workflows to project.")

    for wf in target_workflows:
        print(f"Projecting workflow: {wf['label']}")
        
        # A. Create Phenotype Projection Node
        # This represents the "holographic simulation" or "actualized form" of the workflow
        proj_id = f"phenotype:{wf['id'].split(':')[-1]}"
        proj_node = {
            "id": proj_id,
            "type": "phenotype_projection",
            "label": f"Projected: {wf['label']}",
            "props": {
                "simulation_mode": "holographic",
                "gates": ["P", "E", "L", "R", "Q"]
            },
            "tags": ["phenotype", "simulation", "sota"]
        }
        append_ndjson(KG_PATH, proj_node)
        
        # B. Edge: Workflow -> Phenotype (Projection)
        edge = {
            "from": wf["id"],
            "to": proj_id,
            "rel": "projected_as",
            "props": {
                "gate_P": "potential_verified",
                "gate_E": "entelechy_active",
                "gate_L": "logos_compliant",
                "gate_R": "rheomode_flow",
                "gate_Q": "qualia_witnessed"
            }
        }
        append_ndjson(KG_PATH, edge)
        
        # C. Emit Semantic Event
        semioticalysis = SemioticalysisLayer(
            syntactic="Phenotype Projection",
            semantic=f"Projected '{wf['label']}' into phenotype space via P/E/L/R/Q gates",
            pragmatic="Enabling simulation and validation of the workflow",
            metalinguistic="Sextet Compliance"
        )
        
        entelexis = EntelexisState(
            phase="actualizing",
            form_signature="Phenotype Projection Node",
            material_context="KG Topology",
            actualization_progress=0.5 # Projection is step 1 of actualization
        )
        
        event = create_semantic_event(
            topic="sota.evolution.project",
            kind="state",
            level="info",
            actor="sagent",
            data={
                "workflow_id": wf["id"],
                "projection_id": proj_id,
                "gates": ["P", "E", "L", "R", "Q"]
            },
            semantic=f"Projected workflow '{wf['label']}' to phenotype {proj_id}",
            reasoning="Workflows must be projected to allow for simulation and testing",
            actionable=["Run holographic simulation", "Verify gate compliance"],
            impact="medium",
            semioticalysis=semioticalysis,
            entelexis=entelexis
        )
        
        append_ndjson(BUS_PATH, event.to_dict())
        print(f"  Projected {wf['label']} -> {proj_id}")

if __name__ == "__main__":
    run()
