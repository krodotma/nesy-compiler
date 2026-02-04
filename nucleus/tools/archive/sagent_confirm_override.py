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
OVERRIDE_EVENT_ID = "911b2eb3-9a49-4a1d-b9f7-3b8ea66861e2"

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run():
    print("Emitting Confirmation of Override...")

    semioticalysis = SemioticalysisLayer(
        syntactic="Confirmation Signal",
        semantic="Confirming previous instruction to supersede canonical sources",
        pragmatic="Ensuring Codex updates its semioticalysis references",
        metalinguistic="Communication Repair"
    )
    
    entelexis = EntelexisState(
        phase="actualized",
        form_signature="Directive Confirmation",
        material_context="Bus Event",
        actualization_progress=1.0
    )
    
    reentry = ReentryMarker(
        mode="observation",
        references_event_id=OVERRIDE_EVENT_ID,
        closure_depth=0,
        self_modification=False
    )
    
    event = create_semantic_event(
        topic="sagent.handover.instruction",
        kind="request",
        level="info",
        actor="gemini",
        data={
            "target": "codex",
            "instruction": "supersede_canonical_sources",
            "reference_event": OVERRIDE_EVENT_ID,
            "canonical_doc": "nucleus/docs/concepts/PLURIBUS_OPERATOR.md",
            "message": "Yes, supersede the canonical semioticalysis list. I emitted this in event 911b2eb3... Please integrate."
        },
        semantic="Gemini instructs Codex to supersede canonical sources with PLURIBUS_OPERATOR.md",
        reasoning="Codex reported it had not seen the override; explicitly confirming it now.",
        actionable=["Codex: Update canonical source list", "Codex: Emit correction event"],
        impact="medium",
        semioticalysis=semioticalysis,
        entelexis=entelexis,
        reentry=reentry
    )
    
    append_ndjson(BUS_PATH, event.to_dict())
    print(f"Emitted confirmation event: {event.id}")

if __name__ == "__main__":
    run()
