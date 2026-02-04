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
TRANSCRIPTS_ROOT = Path("/pluribus/transcripts")
DISTILLATIONS_ROOT = Path("/pluribus/nucleus/docs/strp/distillations")

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
    print("Starting Sagent Iteration 3...")

    # 1. Load KG
    kg_items = load_kg_nodes(KG_PATH)
    print(f"Loaded {len(kg_items)} items from KG.")

    # 2. Identify SOTA Items
    sota_items = [item for item in kg_items if item.get("type") == "sota_item"]
    print(f"Found {len(sota_items)} SOTA items.")
    
    # Check for existing artifact links to avoid dupes
    linked_artifacts = set()
    for item in kg_items:
        if item.get("rel") in ["has_transcript", "has_distillation"]:
            linked_artifacts.add(item.get("to"))

    for item in sota_items:
        video_id = item["id"].split(":")[-1] # sota:video:ID
        print(f"Processing SOTA Item: {item['label']} ({video_id})")
        
        # A. Transcript Artifact
        transcript_path = TRANSCRIPTS_ROOT / f"{video_id}.en.txt"
        if transcript_path.exists():
            art_id = f"artifact:transcript:{video_id}"
            if art_id not in linked_artifacts:
                # Node
                art_node = {
                    "id": art_id,
                    "type": "artifact",
                    "label": f"Transcript: {item['label']}",
                    "props": {
                        "path": str(transcript_path),
                        "format": "text/plain",
                        "source_type": "youtube_transcript"
                    },
                    "tags": ["artifact", "transcript", "sota"]
                }
                append_ndjson(KG_PATH, art_node)
                
                # Edge
                edge = {
                    "from": item["id"],
                    "to": art_id,
                    "rel": "has_transcript",
                    "props": {}
                }
                append_ndjson(KG_PATH, edge)
                print(f"  Linked Transcript: {transcript_path}")
                
                # Event
                _emit_artifact_event(item, art_node, "transcript")
        else:
            print(f"  Warning: Transcript not found at {transcript_path}")

        # B. Distillation Artifact
        distill_path = DISTILLATIONS_ROOT / f"{video_id}_distill.md"
        if distill_path.exists():
            art_id = f"artifact:distill:{video_id}"
            if art_id not in linked_artifacts:
                # Node
                art_node = {
                    "id": art_id,
                    "type": "artifact",
                    "label": f"Distillation: {item['label']}",
                    "props": {
                        "path": str(distill_path),
                        "format": "markdown",
                        "source_type": "strp_distillation"
                    },
                    "tags": ["artifact", "distillation", "sota"]
                }
                append_ndjson(KG_PATH, art_node)
                
                # Edge
                edge = {
                    "from": item["id"],
                    "to": art_id,
                    "rel": "has_distillation",
                    "props": {}
                }
                append_ndjson(KG_PATH, edge)
                print(f"  Linked Distillation: {distill_path}")
                
                # Event
                _emit_artifact_event(item, art_node, "distillation")
        else:
             print(f"  Warning: Distillation not found at {distill_path}")

def _emit_artifact_event(sota_item, artifact_node, art_type):
    semioticalysis = SemioticalysisLayer(
        syntactic=f"{art_type.capitalize()} File",
        semantic=f"Physical artifact for '{sota_item['label']}'",
        pragmatic="Grounding knowledge in file system reality",
        metalinguistic="Artifact Binding"
    )
    
    entelexis = EntelexisState(
        phase="actualized",
        form_signature=f"{art_type.capitalize()} Node",
        material_context=artifact_node["props"]["path"],
        actualization_progress=1.0
    )
    
    event = create_semantic_event(
        topic="sota.artifact.bind",
        kind="artifact",
        level="info",
        actor="sagent",
        data={
            "sota_id": sota_item["id"],
            "artifact_id": artifact_node["id"],
            "path": artifact_node["props"]["path"]
        },
        semantic=f"Bound {art_type} for '{sota_item['label']}' to KG",
        reasoning="Connecting semantic nodes to physical evidence",
        actionable=["Read artifact content", "Verify path accessibility"],
        impact="low",
        semioticalysis=semioticalysis,
        entelexis=entelexis
    )
    
    append_ndjson(BUS_PATH, event.to_dict())

if __name__ == "__main__":
    run()
