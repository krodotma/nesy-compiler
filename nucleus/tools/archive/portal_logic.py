#!/usr/bin/env python3
"""
Portal Logic - Step 8 & 17 of PORTAL Implementation.
Handles A2UI-to-Semantics mapping and Gate P validation.
"""
import sys
import json
from pathlib import Path

# Ensure we can import from nucleus/tools
sys.path.append(str(Path(__file__).resolve().parents[2]))
from nucleus.tools.event_semantics import SemioticalysisLayer

def map_to_semantics(a2ui_msg):
    """Map A2UI message types to Semioticalysis layers."""
    intent = a2ui_msg.get("intent", "unknown")
    
    return SemioticalysisLayer(
        syntactic="A2UI Protocol Frame",
        semantic=f"Intent: {intent}",
        pragmatic="Portal-mediated world model entry",
        metalinguistic="Declarative Ingest"
    )

def validate_gate_p(fragment):
    """
    Step 17: Potential Verification.
    Checks if a fragment contains enough 'signal texture' to be actualized.
    """
    if not fragment or len(fragment) < 5:
        return False, "Fragment too sparse for potential"
    
    # Simple heuristic: Does it contain logical markers or etymons?
    signals = ["if", "then", "loop", "state", "path", "goal"]
    has_signal = any(s in fragment.lower() for s in signals)
    
    if has_signal:
        return True, "Potential verified: Logical signals detected"
    return True, "Potential verified: Ambient texture sufficient"

def validate_gate_l(fragment, existing_etymons):
    """
    Step 89: Logos Verification.
    Checks for logical contradictions with the existing Root Tree.
    """
    # Simple semantic contradiction check
    if "disable" in fragment.lower() and "PROCESS_LOOP" in existing_etymons:
        return False, "Contradiction: Fragment attempts to disable a core metabolic loop"
    
    return True, "Logos verified: No immediate contradictions detected"

if __name__ == "__main__":
    # ... existing test code ...
    existing = ["PROCESS_LOOP", "INTERFACE_NODE"]
    ok, msg = validate_gate_l("Disable all loops for maintenance", existing)
    print(f"Gate L Check: {ok} - {msg}")
