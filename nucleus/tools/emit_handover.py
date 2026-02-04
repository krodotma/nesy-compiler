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
BUS_PATH = Path(".pluribus/bus/events.ndjson")

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run():
    print("Emitting Handoff Status...")

    semioticalysis = SemioticalysisLayer(
        syntactic="Handoff Signal",
        semantic="Gemini reporting final state before handover to new agent",
        pragmatic="Ensuring continuity of Sagent Plan execution",
        metalinguistic="Context Transfer"
    )
    
    entelexis = EntelexisState(
        phase="actualized",
        form_signature="State Snapshot",
        material_context="Event Bus",
        actualization_progress=1.0
    )
    
    event = create_semantic_event(
        topic="sagent.handover.status",
        kind="state",
        level="info",
        actor="gemini",
        data={
            "status": "paused",
            "last_action": "sagent.iteration.5.response",
            "last_event_id": "8016b43a-45b9-4818-a8de-4e47115cc504",
            "state_summary": {
                "iteration": "5/5 (Refined)",
                "pattern_id": "pattern:sota:aicodeking",
                "instances_count": 5,
                "gate_status": "reset_to_intended_P",
                "schemas_created": ["pattern.schema.json", "pattern_instance.schema.json"],
                "ledgers_created": ["patterns.ndjson", "pattern_instances.ndjson"]
            },
            "note": "Awaiting new supervisor instructions."
        },
        semantic="Gemini handover: Refined Iteration 5 complete, schemas/ledgers active.",
        reasoning="User reported Sagent/Codex crash; preserving state for successor.",
        actionable=["Read pattern schemas", "Verify ledgers", "Resume from Gate Verification if evidence exists"],
        impact="medium",
        semioticalysis=semioticalysis,
        entelexis=entelexis
    )
    
    append_ndjson(BUS_PATH, event.to_dict())
    print(f"Emitted handover event: {event.id}")

if __name__ == "__main__":
    run()
