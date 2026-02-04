#!/usr/bin/env python3
"""
CMP Engine - Clade Metaproductivity Tracking Daemon (Agent Alpha).
Calculates lineage productivity using temporal discounts and entropy smoothing.

Subscribes to:
- omega.metrics.entropy (legacy compatibility)
- entropy.profile.computed (LENS H* vector)
- omega.guardian.semantic.cycle (semantic state)
- omega.guardian.semantic.stale (stale warnings)
- omega.guardian.semantic.motif_complete (Buchi acceptance bonus)
- hgt.transfer.request (horizontal gene transfer)

DKIN Version: v28
"""
import os
import sys
import json
import time
import uuid
import argparse
from pathlib import Path
from collections import defaultdict

# Import extension functions for entropy/motif processing
try:
    from cmp_extensions import (
        compute_e_factor,
        compute_motif_bonus,
        adaptive_omega_thresholds,
        EntropyVectorCompat,
        classify_cmp,
        semantic_state_to_liveness,
        MOTIF_BONUSES,
    )
    HAS_EXTENSIONS = True
except ImportError:
    HAS_EXTENSIONS = False

# Constants from .clade-manifest.json
PHI = 1.618033988749895

class CMPEngine:
    def __init__(self, bus_dir, poll_interval=2.0, discount=0.9, smoothing=PHI):
        self.bus_dir = Path(bus_dir)
        self.events_path = self.bus_dir / "events.ndjson"
        self.lineages = {}  # lineage_id -> {metrics, reward, parent, cmp_score}
        self.tree = defaultdict(list)  # parent_id -> [child_ids]
        self.last_pos = 0
        self.poll_interval = float(poll_interval)
        self.discount = float(discount)
        self.smoothing = float(smoothing)
        self.running = True
        self.actor = "alpha-cmp"
        self.global_entropy = 0.5
        self.global_liveness = 1.0  # Semantic completion rate from omega guardian
        self.motif_completions = 0  # Total motif completions tracked

    def emit(self, topic, kind, level, data):
        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self.actor,
            "data": data
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")

    def process_event(self, event):
        topic = event.get("topic")
        data = event.get("data", {})
        if not isinstance(data, dict):
            return

        # Legacy entropy topic
        if topic == "omega.metrics.entropy":
            self.global_entropy = float(data.get("topic_entropy", self.global_entropy))
            return

        # LENS entropy profile (canonical topic from lens_entropy_profiler.py)
        if topic == "entropy.profile.computed":
            self._handle_entropy_profile(event)
            return

        # Omega semantic cycle
        if topic == "omega.guardian.semantic.cycle":
            self.global_liveness = float(data.get("completion_rate", self.global_liveness))
            print(f"[CMP] Global liveness updated: {self.global_liveness:.4f}")
            return

        # Omega motif completion (Buchi acceptance)
        if topic == "omega.guardian.semantic.motif_complete":
            self._handle_motif_complete(event)
            return

        # Horizontal gene transfer
        if topic == "hgt.transfer.request":
            self.handle_hgt_request(data)
            return

        # Lineage-specific events
        lineage_id = data.get("lineage_id")
        if not lineage_id:
            return

        if lineage_id not in self.lineages:
            parent_id = data.get("parent_lineage_id")
            self.lineages[lineage_id] = {
                "reward": 0.0,
                "parent": parent_id,
                "cmp_score": 0.0,
                "events_count": 0,
                "entropy_profile": {},
                "quality_functional": 0.0,
                "liveness_penalty": 0.0,
                "hgt_count": 0,
                "motif_bonus": 0.0,
                "motifs_completed": 0,
            }
            if parent_id:
                self.tree[parent_id].append(lineage_id)
            print(f"[CMP] Registered lineage: {lineage_id} (parent: {parent_id})")

        l = self.lineages[lineage_id]
        l["events_count"] += 1

        if topic == "omega.guardian.semantic.stale" and data.get("target_actor"):
            l["liveness_penalty"] += 0.1

        if "reward" in data:
            l["reward"] = float(data["reward"])

        if "entropy_profile" in data:
            l["entropy_profile"].update(data["entropy_profile"])
        elif "entropy_score" in data:
            l["entropy_profile"]["qa_score"] = float(data["entropy_score"])

        self.update_cmp(lineage_id)

    def _handle_entropy_profile(self, event):
        """Handle entropy.profile.computed events from LENS."""
        data = event.get("data", {})
        entropy_vector = data.get("entropy_vector", {})

        # Update global entropy from H* vector
        if entropy_vector:
            h_mean = entropy_vector.get("h_mean", 0.5)
            h_info = entropy_vector.get("h_info", 0.5)
            # Global entropy = average of failure entropies
            self.global_entropy = h_mean

            # Compute E-factor if extensions available
            if HAS_EXTENSIONS:
                e_factor = compute_e_factor(entropy_vector)
                print(f"[CMP] LENS entropy profile: h_info={h_info:.3f}, h_mean={h_mean:.3f}, e_factor={e_factor:.3f}")
            else:
                print(f"[CMP] LENS entropy profile: h_info={h_info:.3f}, h_mean={h_mean:.3f}")

    def _handle_motif_complete(self, event):
        """Handle omega.guardian.semantic.motif_complete events."""
        data = event.get("data", {})
        motif_id = data.get("motif_id", "unknown")
        actor = data.get("event_actor") or data.get("actor") or event.get("actor", "unknown")
        duration_s = data.get("duration_s", 60.0)
        weight = data.get("weight", 1.0)

        self.motif_completions += 1

        # Boost global liveness on motif completion
        self.global_liveness = min(1.0, self.global_liveness + 0.05)

        # Compute bonus if extensions available
        if HAS_EXTENSIONS:
            bonus = compute_motif_bonus(event, f"actor.{actor}")
        else:
            # Fallback: fixed bonus
            bonus = 0.05 * min(2.0, weight)

        # Find and update relevant lineages for this actor
        for lineage_id, l in self.lineages.items():
            if actor in lineage_id or lineage_id.startswith(f"actor.{actor}"):
                l["motif_bonus"] += bonus
                l["motifs_completed"] += 1
                self.update_cmp(lineage_id)
                print(f"[CMP] Motif {motif_id} by {actor}: bonus={bonus:.4f} -> lineage {lineage_id}")

        # Emit aggregate event
        self.emit("cmp.motif.processed", "metric", "info", {
            "motif_id": motif_id,
            "actor": actor,
            "bonus": bonus,
            "global_liveness": self.global_liveness,
            "total_motifs": self.motif_completions,
        })

    def handle_hgt_request(self, data):
        source_id = data.get("source_lineage_id")
        target_id = data.get("target_lineage_id")
        motif_id = data.get("motif_id")

        if not source_id or not target_id or source_id not in self.lineages or target_id not in self.lineages:
            return

        source = self.lineages[source_id]
        target = self.lineages[target_id]

        # Compatibility Check: Entropy Delta
        s_uy = source["quality_functional"]
        t_uy = target["quality_functional"]

        # Transfer logic: if source is significantly better or highly compatible
        compatibility = 1.0 - abs(s_uy - t_uy)
        transfer_potential = (source["cmp_score"] / max(0.001, target["cmp_score"])) * compatibility

        decision = "approve" if transfer_potential > PHI / 2.0 else "reject"

        if decision == "approve":
            target["hgt_count"] += 1
            # Boost target score slightly for successful gene acquisition
            target["reward"] += 0.05
            self.update_cmp(target_id)

        self.emit("cmp.hgt.decision", "response", "info", {
            "source_lineage_id": source_id,
            "target_lineage_id": target_id,
            "motif_id": motif_id,
            "decision": decision,
            "transfer_potential": transfer_potential,
            "compatibility": compatibility
        })
        print(f"[CMP] HGT Decision: {decision} ({source_id} -> {target_id}) potential: {transfer_potential:.4f}")

    def calculate_uy(self, lineage_id):
        l = self.lineages[lineage_id]
        ep = l["entropy_profile"]
        h_info = ep.get("h_info", 1.0)
        h_miss = ep.get("h_miss", 0.0)
        h_conj = ep.get("h_conj", 0.0)
        h_alea = ep.get("h_alea", 0.0)
        c_load = ep.get("c_load", 0.0)
        qa_score = ep.get("qa_score", 0.0)

        failures = [h_miss, h_conj, h_alea, qa_score, l["liveness_penalty"]]
        u_y = h_info
        for f in failures:
            u_y *= (1.0 - min(1.0, f))
        u_y /= (1.0 + c_load)

        # Add motif bonus to quality functional
        u_y += l.get("motif_bonus", 0.0)

        l["quality_functional"] = u_y
        return u_y

    def update_cmp(self, lineage_id):
        l = self.lineages[lineage_id]
        uy = self.calculate_uy(lineage_id)

        # Spectral Smoothing Implementation:
        # Instead of just current reward, we mix in the Laplacian contribution
        # (weighted average of neighbors/siblings) to regularize the clade score.
        score = l["reward"] * uy * self.global_liveness

        children = self.tree.get(lineage_id, [])
        child_contribution = 0.0
        for cid in children:
            child_contribution += self.lineages[cid]["cmp_score"]

        if children:
            # Recursive Clade Metaproductivity
            score = (score + self.discount * (child_contribution / len(children))) / self.smoothing

        # Final Spectral Mix with parent context if available
        if l["parent"] and l["parent"] in self.lineages:
            parent_score = self.lineages[l["parent"]]["cmp_score"]
            score = 0.8 * score + 0.2 * parent_score # mix in 20% parent stability

        # Step 127: Epigenetic Lift (Shadow -> Light contribution)
        if l.get("hgt_count", 0) > 0:
            lift = (l["hgt_count"] * 0.05) / max(1, l["events_count"])
            score *= (1.0 + lift)
            print(f"[CMP] Lineage {lineage_id} received Epigenetic Lift: +{lift*100:.1f}%")

        if abs(l["cmp_score"] - score) > 0.0001:

        
            l["cmp_score"] = score
            print(f"[CMP] New score for {lineage_id}: {score:.4f} (UY: {uy:.4f})")
            self.emit("cmp.lineage.update", "metric", "info", {
                "lineage_id": lineage_id,
                "cmp_score": score,
                "u_y": uy,
                "global_liveness": self.global_liveness,
                "parent_id": l["parent"],
                "motifs_completed": l.get("motifs_completed", 0),
            })
            if l["parent"] and l["parent"] in self.lineages:
                self.update_cmp(l["parent"])
    

    def run(self, bootstrap=False):
        print(f"CMP Engine starting. Watching {self.events_path}")
        if self.events_path.exists():
            if bootstrap:
                self.last_pos = 0
            else:
                self.last_pos = self.events_path.stat().st_size

        while self.running:
            if not self.events_path.exists():
                time.sleep(self.poll_interval)
                continue
            with self.events_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self.last_pos)
                lines = f.readlines()
                self.last_pos = f.tell()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    try:
                        event = json.loads(line)
                        self.process_event(event)
                    except:
                        continue
            time.sleep(self.poll_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bus-dir", default="/pluribus/.pluribus/bus")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--smoothing", type=float, default=PHI)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    engine = CMPEngine(args.bus_dir, poll_interval=args.poll_interval, 
                       discount=args.discount, smoothing=args.smoothing)
    try:
        engine.run(bootstrap=args.bootstrap)
    except KeyboardInterrupt:
        print("CMP Engine stopped.")