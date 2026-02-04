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

BUS_PATH = Path(".pluribus/bus/events.ndjson")

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run():
    print("Emitting PBLOCK Request...")

    semioticalysis = SemioticalysisLayer(
        syntactic="PBLOCK Signal",
        semantic="Requesting Pipeline Block (Milestone Freeze)",
        pragmatic="Stabilizing state after Sagent Integration",
        metalinguistic="Phase Transition"
    )
    
    entelexis = EntelexisState(
        phase="potential",
        form_signature="Frozen State",
        material_context="Git/Bus",
        actualization_progress=0.0
    )
    
    event = create_semantic_event(
        topic="operator.pblock.request",
        kind="request",
        level="info",
        actor="gemini",
        data={
            "milestone": "sagent_integration_v1",
            "reason": "Completed 5-iteration Sagent Plan + Verification",
            "verification_status": "200% Verified (PBCLITEST)",
            "dirty_state": "pending_commit"
        },
        semantic="Gemini requests PBLOCK: Sagent Integration complete and verified.",
        reasoning="Milestone achieved; freezing state for stability.",
        actionable=["Review pending commit", "Ack PBLOCK"],
        impact="high",
        semioticalysis=semioticalysis,
        entelexis=entelexis
    )
    
    append_ndjson(BUS_PATH, event.to_dict())
    print(f"Emitted PBLOCK request: {event.id}")

if __name__ == "__main__":
    run()
