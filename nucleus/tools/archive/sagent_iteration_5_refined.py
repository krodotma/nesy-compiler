#!/usr/bin/env python3
import json
import sys
import uuid
import time
from pathlib import Path

# Ensure we can import from nucleus/tools
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))

from nucleus.tools.event_semantics import create_semantic_event, EntelexisState, SemioticalysisLayer, ReentryMarker

# Configuration
PATTERNS_PATH = Path(".pluribus/index/patterns.ndjson")
INSTANCES_PATH = Path(".pluribus/index/pattern_instances.ndjson")
BUS_PATH = Path(".pluribus/bus/events.ndjson")
KG_PATH = Path(".pluribus/index/kg_nodes.ndjson")

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
    print("Starting Sagent Iteration 5 Refinement (Compliance)...")

    # 1. Define Pattern
    pattern_id = "pattern:sota:aicodeking"
    pattern = {
        "id": pattern_id,
        "family": "sota_distillation",
        "name": "AICodeKing Ingest",
        "description": "Ingest pattern for AICodeKing SOTA videos to KG",
        "teleology": {
            "purpose": "Accelerate developer velocity via AI-assisted workflows",
            "goals": ["Mastery of SOTA tools", "Integration of multi-model pipelines"],
            "values": ["Speed", "Efficiency", "Automation"]
        },
        "gates_intended": ["P"] # Per decision lock
    }
    append_ndjson(PATTERNS_PATH, pattern)
    print(f"Registered pattern: {pattern_id}")

    # 2. Register Instances (correcting previous gate claims)
    kg_items = load_kg_nodes(KG_PATH)
    projections = [item for item in kg_items if item.get("type") == "phenotype_projection"]
    
    instance_ids = []
    
    for proj in projections:
        # Infer workflow ID from projection ID (reverse engineering our naming convention)
        # proj_id = phenotype:ID, workflow_id = workflow:ID
        vid_id = proj["id"].split(":")[-1]
        workflow_id = f"workflow:{vid_id}"
        
        instance_id = f"instance:{uuid.uuid4()}"
        instance = {
            "id": instance_id,
            "pattern_id": pattern_id,
            "workflow_id": workflow_id,
            "projection_id": proj["id"],
            "status": "active",
            "gates_verified": [] # Per decision lock: reset to empty until evidence
        }
        append_ndjson(INSTANCES_PATH, instance)
        instance_ids.append(instance_id)
        print(f"  Registered instance for {workflow_id}")

    # 3. Emit Response Event
    semioticalysis = SemioticalysisLayer(
        syntactic="Compliance Response",
        semantic="Acknowledging Sagent Iteration 5 decisions and correcting gate state",
        pragmatic="Aligning with locked decision logic and schemas",
        metalinguistic="Correction & Alignment"
    )
    
    entelexis = EntelexisState(
        phase="actualizing", # Downgraded from actualized until gates verified
        form_signature="Pattern Instance Ledger",
        material_context="Pattern Schemas & Ledgers",
        actualization_progress=0.8
    )
    
    reentry = ReentryMarker(
        mode="modification",
        references_event_id="sagent.iteration.5", # Implicit ref to supervisor's locked decision
        closure_depth=1,
        self_modification=True
    )
    
    event = create_semantic_event(
        topic="sagent.iteration.5.response",
        kind="response",
        level="info",
        actor="gemini",
        data={
            "status": "aligned",
            "compliance": {
                "schema_files": ["pattern.schema.json", "pattern_instance.schema.json"],
                "ledgers": ["patterns.ndjson", "pattern_instances.ndjson"],
                "gate_discipline": "reset_to_intended_P",
                "link_rules": "pattern_id_added"
            },
            "pattern_id": pattern_id,
            "instances_created": len(instance_ids),
            "correction_note": "Supersedes previous sagent.iteration.5.complete event regarding gate verification."
        },
        semantic=f"Gemini aligned with Sagent Iteration 5 locks. Registered pattern '{pattern_id}' and reset gate verification.",
        reasoning="Supervisor locked decisions require strict schema adherence and evidence-based gate verification.",
        actionable=["Await evidence for Gate P verification", "Monitor pattern instances"],
        impact="medium",
        semioticalysis=semioticalysis,
        entelexis=entelexis,
        reentry=reentry
    )
    
    append_ndjson(BUS_PATH, event.to_dict())
    print(f"Emitted response event: {event.id}")

if __name__ == "__main__":
    run()
