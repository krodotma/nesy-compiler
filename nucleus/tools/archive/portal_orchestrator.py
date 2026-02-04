#!/usr/bin/env python3
"""
Portal Ingest Orchestrator - Step 21 & 25 of PORTAL Implementation.
Orchestrates the metabolic fork between Entelexis (AM) and Hysteresis (SM).
"""
import sys
import json
import uuid
import time
from pathlib import Path

# Setup paths
sys.path.append(str(Path(__file__).resolve().parents[2]))
from nucleus.tools.portal_denoiser import denoise_fragment
from nucleus.tools.portal_decoder import UniversalSemanticDecoder
from nucleus.tools.portal_logic import validate_gate_p

class PortalOrchestrator:
    def __init__(self, current_goal="implement portal", bus_path=".pluribus/bus/events.ndjson"):
        self.current_goal = current_goal
        self.decoder = UniversalSemanticDecoder()
        self.bus_path = Path(bus_path)
        self.noise_floor = 0.01 # Threshold lowered for testing

    def process_ingest(self, raw_input, lineage_id="trunk"):
        # 1. Denoise
        clean = denoise_fragment(raw_input)
        
        # 2. Validate Potential (Gate P)
        ok, msg = validate_gate_p(clean)
        if not ok:
            return {"status": "rejected", "reason": msg}
            
        # 3. Decode Texture
        decoded = self.decoder.decode(clean)
        
        # Step 35: Noise Floor Pruning
        if decoded["texture_density"] < self.noise_floor:
            return {"status": "pruned", "reason": "Signal below noise floor"}

        # 4. Metabolic Fork Logic (The Rheomode Decision)
        # Is the process actualizing the goal?
        is_relevant = any(w in clean.lower() for w in self.current_goal.split())
        
        if is_relevant:
            mode = "AM" # Actualized-Mode (Conscious)
            topic = "entelexis.actualize"
            reward = 0.1 # Boost for Step 32
        else:
            mode = "SM" # Shadowmode (Subconscious)
            topic = "hysteresis.shadow"
            reward = 0.02 # Minimal boost for shadow learning
            
        # Step 86: Multimodal Generation Bridge
        if "video" in clean.lower() or "simulate" in clean.lower():
            self.emit_gen3c_request(decoded)

        # Step 32: Emit CMP Signal
        self.emit_cmp_signal(lineage_id, reward, decoded["texture_density"])

        return {
            "mode": mode,
            "topic": topic,
            "decoded": decoded,
            "signal": clean
        }

    def emit_gen3c_request(self, decoded):
        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": "sota.gen3c.request",
            "kind": "request",
            "level": "info",
            "actor": "agent.portal",
            "data": {
                "req_id": f"gen3c-{uuid.uuid4().hex[:8]}",
                "texture": decoded,
                "goal": self.current_goal,
                "duration": 4
            }
        }
        with self.bus_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def emit_cmp_signal(self, lineage_id, reward, entropy):
        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": "cmp.lineage.update",
            "kind": "metric",
            "level": "info",
            "actor": "agent.portal",
            "data": {
                "lineage_id": lineage_id,
                "reward": reward,
                "entropy_profile": {"h_info": 1.0 - entropy}
            }
        }
        with self.bus_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

if __name__ == "__main__":
    orchestrator = PortalOrchestrator()
    result = orchestrator.process_ingest(sys.argv[1] if len(sys.argv) > 1 else "empty")
    print(json.dumps(result, indent=2))
