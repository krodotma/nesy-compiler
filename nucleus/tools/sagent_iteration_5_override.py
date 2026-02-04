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
NEW_CANONICAL_DOC = "nucleus/docs/concepts/PLURIBUS_OPERATOR.md"

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run():
    print("Emitting Canonical Source Override...")

    semioticalysis = SemioticalysisLayer(
        syntactic="Canonical Override",
        semantic="Superseding semioticalysis sources with PLURIBUS_OPERATOR.md",
        pragmatic="Aligning system interpretation with the formal PLURIBUS modality operator",
        metalinguistic="SemIps Re-binding"
    )
    
    entelexis = EntelexisState(
        phase="actualized",
        form_signature="Canonical Source Configuration",
        material_context="Nucleus Docs",
        actualization_progress=1.0
    )
    
    reentry = ReentryMarker(
        mode="self_modification",
        references_event_id="sagent.iteration.5", # Responding to the supervisor's lock
        closure_depth=2,
        self_modification=True
    )
    
    event = create_semantic_event(
        topic="sagent.iteration.5.response",
        kind="response",
        level="info",
        actor="gemini",
        data={
            "status": "superseded",
            "canonical_sources_override": {
                "primary": NEW_CANONICAL_DOC,
                "secondary": [
                    "nucleus/tools/event_semantics.py",
                    "nucleus/docs/strp/schemas/dimensional_event.schema.json",
                    "nucleus/specs/pluribus_lexicon.md"
                ],
                "note": "PLURIBUS_OPERATOR.md is now the authoritative source for PLURIBUS modality and SemIps pipeline."
            },
            "interpretation_alignment": [
                "PLURIBUS as first-class semantic operator token",
                "SemIps interpretation function (Σ* ⇀ (IntentGraph, EvidencePlan))",
                "Intent DAG as KG subgraph",
                "AuOM laws and Entelexis phase semantics"
            ]
        },
        semantic="Gemini confirms override: PLURIBUS_OPERATOR.md is now canonical.",
        reasoning="Formalizing the PLURIBUS operator as a typed modality is superior to basic schema-based definitions.",
        actionable=["Apply SemIps logic to future prompts", "Monitor Intent DAG emissions"],
        impact="high",
        semioticalysis=semioticalysis,
        entelexis=entelexis,
        reentry=reentry
    )
    
    append_ndjson(BUS_PATH, event.to_dict())
    print(f"Emitted override event: {event.id}")

if __name__ == "__main__":
    run()
