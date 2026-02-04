#!/usr/bin/env python3
"""
Universal Semantic Decoder - Step 23 of PORTAL Implementation.
Decodes raw ingest fragments into structured Etymonic forms.
"""
import sys
import json

class UniversalSemanticDecoder:
    def decode(self, cleaned_text):
        """
        Denoises and decodes text into a 'Texture Signature'.
        This follows the CV lesson: texture = depth.
        """
        # 1. Identify primary logical etymons
        etymons = []
        if any(w in cleaned_text.lower() for w in ["loop", "recur", "iterate", "process"]):
            etymons.append("PROCESS_LOOP")
        if any(w in cleaned_text.lower() for w in ["if", "gate", "decision", "logic"]):
            etymons.append("DECISION_GATE")
        if any(w in cleaned_text.lower() for w in ["graft", "link", "connect", "implement"]):
            etymons.append("GRAPH_EDGE")
        if any(w in cleaned_text.lower() for w in ["ui", "renderer", "dashboard", "portal"]):
            etymons.append("INTERFACE_NODE")
            
        # 2. Wisdom Categories (Step 30)
        categories = []
        if any(w in cleaned_text.lower() for w in ["always", "never", "must"]):
            categories.append("FACT")
        if any(w in cleaned_text.lower() for w in ["often", "regularly", "pattern"]):
            categories.append("HABIT")
        if any(w in cleaned_text.lower() for w in ["maybe", "perhaps", "could"]):
            categories.append("INSIGHT")

        # 3. Calculate Texture Density
        # (High density = rich in logical instructions)
        density = (len(etymons) + len(categories)) / max(1, len(cleaned_text.split()))
        
        return {
            "etymons": etymons,
            "categories": categories,
            "texture_density": density,
            "original_signal": cleaned_text[:100] + "..."
        }

if __name__ == "__main__":
    usd = UniversalSemanticDecoder()
    print(json.dumps(usd.decode(sys.argv[1] if len(sys.argv) > 1 else "")))
