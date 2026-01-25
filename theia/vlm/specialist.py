"""
Theia VLM Specialist — Core Vision-Language Agent Logic.

Integrates all Theia layers into a cohesive specialist:
    L0: Capture
    L1: Geometric embedding
    L2: Memory retrieval (mHC)
    L3: Crystallization
    L4: Automata execution
    L5: Metacognition
    L8: Program synthesis (ICL)
    L10: Swarm collaboration (DKIN v18)

This is the main entry point for "Theia" as an agent behavior.
"""

import time
import base64
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Layer imports
from theia.capture import WindowCapture
from theia.geometric.fiber_bundle import FiberBundle
from theia.memory import HopfieldMemory, create_memory, ARCPrescienceEngine
from theia.memory.temporal_sync import create_hierarchy_viz
from theia.crystallize import BirkhoffPolytope
from theia.automata import DNAAutomaton, SelfTeacher
from theia.meta import OmegaDomain, Metacognition
from theia.synthesis import EGGPEvolver
from theia.vlm.icl import ScreenshotICL, ScreenshotExample
from theia.swarm import SwarmDispatcher, TaskLifecycle

from theia.governance import GuardLadder, GuardResult

@dataclass
class TheiaConfig:
    """Configuration for Theia specialist."""
    capture_index: int = 1
    memory_capacity: int = 1000
    prescience_scales: int = 3
    automaton_states: int = 5
    enable_metacognition: bool = True
    vps_mode: bool = True  # Optimized for VPS execution (headless)
    enable_swarm: bool = True  # DKIN v18 task lifecycle

class VLMSpecialist:
    """
    Theia VLM Specialist.
    
    The "Eye of the System" that provides:
    1. Real-time visual understanding
    2. Self-teaching automation
    3. ARC-level prescience
    4. Swarm collaboration (DKIN v29)
    5. Action Governance (Guard Ladder G1-G6)
    """
    
    def __init__(self, config: TheiaConfig = TheiaConfig()):
        self.config = config
        self.active = False
        self._boot_time = time.time()
        
        # --- L0: Perception ---
        self.capture = WindowCapture(monitor_index=config.capture_index)
        
        # --- L1: Geometry ---
        self.fiber_bundle = FiberBundle(base_dim=32, fiber_dim=32)
        
        # --- L2: Memory & Time ---
        self.memory = create_memory(dimension=64)
        self.prescience = ARCPrescienceEngine(n_scales=config.prescience_scales)
        self.icl = ScreenshotICL()
        
        # --- L3: Crystallization ---
        self.polytope = BirkhoffPolytope(n=4)
        
        # --- L4: Automata ---
        self.automaton = DNAAutomaton()
        self.teacher = SelfTeacher()
        
        # --- L5: Meta ---
        self.omega = OmegaDomain()
        self.metacog = Metacognition(self.omega)
        
        # --- L8: Synthesis ---
        self.eggp = EGGPEvolver()
        self.eggp.initialize()
        
        # --- Governance (DUALITY-BIND) ---
        self.guard = GuardLadder(agent_id="theia")
        
        # --- L10: Swarm (DKIN v29) ---
        if config.enable_swarm:
            self.dispatcher = SwarmDispatcher()
            # TaskLifecycle deprecated, moved to A2AProtocol
            self.lifecycle = TaskLifecycle(agent_id="theia") 
            from theia.swarm import A2AProtocol
            self.a2a = A2AProtocol(agent_id="theia")
            self._boot_task_id: Optional[str] = None
        else:
            self.dispatcher = None
            self.lifecycle = None
            self.a2a = None

    def act(self, intention: str) -> Dict[str, Any]:
        """Execute intention via automata and EGGP (guarded)."""
        program = self.eggp.evolve_generation()
        
        # Formulate action for guard check
        action_candidate = {
            "type": "automata_step",
            "target": "DNAAutomaton",
            "payload": {
                "intention": intention,
                "program_id": program.id
            }
        }
        
        # G1-G6 Validation
        outcomes = self.guard.check(action_candidate)
        failures = [o for o in outcomes if o.level == GuardResult.FAIL]
        
        if failures:
             print(f"[Theia] Action BLOCKED by Guard: {failures[0].message}")
             return {
                 "status": "blocked",
                 "reason": failures[0].message,
                 "id": failures[0].guard_id
             }
        
        # Step automaton
        state_transition = self.automaton.step(environment=hash(intention) % 5)
        
        # Reentry/Self-Teaching
        success = True 
        self.teacher.record_episode([], [intention], success)
        
        return {
            "action": intention,
            "program_id": program.id,
            "automaton_state": self.automaton.current,
            "learning_applied": self.teacher.get_dashboard_data(),
            "guard_outcomes": [o.message for o in outcomes]
        }


    def boot(self) -> Dict[str, Any]:
        """Initialize all subsystems with task lifecycle."""
        print(f"[Theia] Booting on {self.config.vps_mode and 'VPS' or 'Local'}...")
        self.active = True
        
        # Start boot task (DKIN v18)
        if self.lifecycle:
            self._boot_task_id = self.lifecycle.start(
                description="Theia VLM boot sequence",
                context={"config": self.config.__dict__}
            )
        
        # Lift self to Omega domain
        self.omega.lift(self, lambda x: x.status_report())
        
        # Complete boot task
        if self.lifecycle and self._boot_task_id:
            self.lifecycle.complete(self._boot_task_id, result="boot_success")
        
        return {
            "status": "active",
            "layers": 10,  # Now includes swarm
            "boot_time": self._boot_time,
            "swarm_enabled": self.config.enable_swarm,
        }


    def perceive(self) -> Dict[str, Any]:
        """
        Main perception cycle.
        
        1. Capture frame
        2. Embed (L1)
        3. Store/Retrieve (L2)
        4. Update Prescience (L2+)
        5. Check Metacognition (L5)
        """
        # 1. Capture
        # In actual usage this gets a screenshot. For now checking if available.
        try:
            frame = self.capture.capture_frame()
            has_input = frame is not None
        except Exception:
            has_input = False
            
        # 2. Mock embedding for loop continuity if no frame
        embedding = np.random.randn(64) # Placeholder for actual vision encoder output
        
        # 3. Memory Interaction
        # Check if we recognize this state
        retrieved = self.memory.retrieve(embedding)
        
        # 4. Prescience
        sync_state = self.prescience.ingest(embedding)
        
        # 5. Metacognition
        meta_state = self.metacog.reflect(sync_state)
        
        return {
            "has_input": has_input,
            "embedding_norm": float(np.linalg.norm(embedding)),
            "prescience": sync_state,
            "metacog": meta_state.to_dict() if hasattr(meta_state, 'to_dict') else str(meta_state),
        }

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "active": self.active,
            "uptime": time.time() - self._boot_time,
            "memory_items": len(self.memory.patterns) if self.memory.patterns is not None else 0,
            "prescience_ready": self.prescience.hierarchy.scales[0].compute_coherence() > 0.7,
            "automata_state": self.automaton.current,
            "vps_mode": self.config.vps_mode
        }

    def shutdown(self):
        """Clean shutdown."""
        self.active = False
        print("[Theia] Shutdown complete.")

__all__ = ["VLMSpecialist", "TheiaConfig"]
