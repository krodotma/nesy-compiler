#!/usr/bin/env python3
"""
integration.py - ARK Integration with Pluribus Ecosystem

Wires ARK to:
- PBTSO swarm orchestrator
- OHM monitoring
- LASER/LENS pipeline
- Agent Bus events
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("ARK.Integration")


@dataclass
class BusEvent:
    """Event for the Pluribus agent bus."""
    topic: str
    payload: Dict
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "ark"


class ArkBusIntegration:
    """
    Integration with Pluribus Agent Bus.
    
    Emits ARK lifecycle events:
    - ark.commit.proposed
    - ark.commit.accepted
    - ark.commit.rejected
    - ark.distill.started
    - ark.distill.completed
    - ark.genesis (inception)
    """
    
    def __init__(self, bus_path: Optional[str] = None):
        self.bus_path = Path(bus_path) if bus_path else Path(".pluribus/bus/events.ndjson")
    
    def emit(self, topic: str, payload: Dict) -> None:
        """Emit an event to the bus."""
        event = BusEvent(topic=topic, payload=payload)
        
        try:
            self.bus_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.bus_path, 'a') as f:
                f.write(json.dumps({
                    "topic": event.topic,
                    "payload": event.payload,
                    "timestamp": event.timestamp,
                    "source": event.source
                }) + "\n")
            
            logger.debug(f"📡 Emitted: {topic}")
            
        except Exception as e:
            logger.warning(f"Failed to emit bus event: {e}")
    
    def emit_commit_proposed(self, sha: str, message: str, etymology: str) -> None:
        """Emit commit proposed event."""
        self.emit("ark.commit.proposed", {
            "sha": sha,
            "message": message,
            "etymology": etymology
        })
    
    def emit_commit_accepted(self, sha: str, cmp: float, entropy: Dict) -> None:
        """Emit commit accepted event."""
        self.emit("ark.commit.accepted", {
            "sha": sha,
            "cmp": cmp,
            "entropy": entropy
        })
    
    def emit_commit_rejected(self, reason: str, gate: str) -> None:
        """Emit commit rejected event."""
        self.emit("ark.commit.rejected", {
            "reason": reason,
            "gate": gate
        })
    
    def emit_distill_completed(self, source: str, target: str, genes: int, cmp: float) -> None:
        """Emit distillation completed event."""
        self.emit("ark.distill.completed", {
            "source": source,
            "target": target,
            "genes_created": genes,
            "total_cmp": cmp
        })
    
    def emit_genesis(self, inception_id: str, path: str) -> None:
        """Emit genesis/inception event."""
        self.emit("ark.genesis", {
            "inception_id": inception_id,
            "path": path
        })


class OHMIntegration:
    """
    Integration with OHM (Omega Heart Monitor).
    
    Reports ARK health metrics for system-wide monitoring.
    """
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
    
    def report_health(self, repo_path: str) -> Dict:
        """Generate health report for OHM."""
        from nucleus.ark.core.repository import ArkRepository
        
        repo = ArkRepository(repo_path)
        status = repo.status()
        
        health = {
            "component": "ark",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy" if status.get("initialized") else "uninitialized",
            "metrics": {
                "gates_enabled": status.get("gates_enabled", False),
                "modified_files": len(status.get("modified", [])),
                "staged_files": len(status.get("staged", [])),
                "entropy": status.get("current_entropy", {})
            }
        }
        
        self.metrics = health["metrics"]
        return health
    
    def get_entropy_summary(self) -> Dict[str, str]:
        """Get human-readable entropy summary."""
        entropy = self.metrics.get("entropy", {})
        summary = {}
        
        for key, value in entropy.items():
            if value < 0.3:
                summary[key] = "low"
            elif value < 0.6:
                summary[key] = "medium"
            else:
                summary[key] = "high"
        
        return summary


class LASERIntegration:
    """
    Integration with LASER/LENS pipeline.
    
    LASER = Language Augmented Superpositional Effective Retrieval
    LENS = LLM Entropic Natural Superposition
    
    Provides 8-dimensional H* entropy vector:
    - h_info: Information density (prompt-relevant signal)
    - h_miss: Missing information (omitted required data)
    - h_conj: Conjecture entropy (unsupported assertions)
    - h_alea: Aleatoric uncertainty (sampling stochasticity)
    - h_epis: Epistemic uncertainty (model knowledge gaps)
    - h_struct: Structural entropy (rhetorical overhead)
    - c_load: Cognitive load (human processing cost)
    - h_goal_drift: Goal drift (divergence from intent)
    
    Also computes utility: U(Y) = h_info * PROD(1 - H_i) / (1 + c_load)
    """
    
    def __init__(self, laser_path: Optional[str] = None):
        self.laser_available = False
        self.EntropyVector = None
        self.profile_entropy = None
        
        # Try to import LASER components from pluribus_evolution
        try:
            import sys
            # Add pluribus_evolution to path if needed
            laser_base = Path("/Users/kroma/pluribus_evolution/laser")
            if laser_base.exists() and str(laser_base.parent) not in sys.path:
                sys.path.insert(0, str(laser_base.parent))
            
            from laser.entropy_profiler import EntropyVector, profile_entropy
            self.EntropyVector = EntropyVector
            self.profile_entropy = profile_entropy
            self.laser_available = True
            logger.info("LASER entropy profiler available - using 8-dimensional H* vector")
        except ImportError as e:
            logger.info("LASER not available - using ARK-native entropy: %s", e)
    
    def profile_entropy(self, code: str, prompt: str = "") -> Dict[str, float]:
        """
        Profile code/text entropy using LASER if available.
        
        Returns 8-dimensional H* vector.
        Falls back to ARK-native calculation otherwise.
        """
        if self.laser_available and self.profile_entropy:
            try:
                vector = self.profile_entropy(Y=code, X=prompt, emit_event=False)
                return vector.to_dict()
            except Exception as e:
                logger.debug("LASER profile failed, using fallback: %s", e)
        
        # Fallback: ARK-native simplified entropy calculation
        return self._native_entropy(code)
    
    def _native_entropy(self, code: str) -> Dict[str, float]:
        """
        ARK-native fallback entropy calculation.
        
        Simpler than LASER but provides compatible H* structure.
        """
        lines = code.split('\n') if code else []
        
        # h_struct: Indentation variance as proxy for structural complexity
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        h_struct = min(len(set(indents)) / 10, 1.0) if indents else 0.5
        
        # h_info: Compression ratio as information density proxy
        if code:
            import zlib
            original = code.encode('utf-8')
            compressed = zlib.compress(original, level=9)
            h_info = len(compressed) / len(original) if len(original) > 0 else 0.5
        else:
            h_info = 0.5
        
        # Other dimensions estimated
        word_count = len(code.split()) if code else 0
        h_doc = 0.3 if '"""' in code or "'''" in code or '#' in code else 0.7
        h_type = 0.3 if ':' in code and '->' in code else 0.6  # Type hints
        h_test = 0.3 if 'test' in code.lower() or 'assert' in code else 0.6
        h_deps = min(code.count('import ') / 20, 1.0)
        h_churn = 0.5  # Would need git history
        h_debt = 0.4 if 'TODO' in code or 'FIXME' in code else 0.3
        h_align = 0.5  # Would need spec comparison
        
        return {
            "h_info": round(h_info, 4),
            "h_miss": 0.5,  # Requires schema
            "h_conj": 0.3,  # Requires claim extraction
            "h_alea": 0.3,  # Requires resampling
            "h_epis": 0.3,  # Requires multi-model
            "h_struct": round(h_struct, 4),
            "c_load": round(min(word_count / 500, 1.0), 4),
            "h_goal_drift": 0.3,  # Requires prompt comparison
            "h_total": round(h_struct + h_info + h_doc, 4),
            "utility": round((1 - h_struct) * h_info, 4),
            "_source": "ark-native"
        }
    
    def compute_utility(self, entropy: Dict[str, float]) -> float:
        """
        Compute utility using LASER formula.
        
        U(Y) = h_info * PROD(1 - H_i) / (1 + c_load)
        """
        h_info = entropy.get("h_info", 0.5)
        h_miss = entropy.get("h_miss", 0.5)
        h_conj = entropy.get("h_conj", 0.3)
        h_alea = entropy.get("h_alea", 0.3)
        h_epis = entropy.get("h_epis", 0.3)
        h_struct = entropy.get("h_struct", 0.5)
        c_load = entropy.get("c_load", 0.3)
        h_goal_drift = entropy.get("h_goal_drift", 0.3)
        
        product = (
            (1.0 - h_miss) *
            (1.0 - h_conj) *
            (1.0 - h_alea) *
            (1.0 - h_epis) *
            (1.0 - h_struct) *
            (1.0 - h_goal_drift)
        )
        
        utility = h_info * product / (1.0 + c_load)
        return max(0.0, min(1.0, utility))


class PBTSOIntegration:
    """
    Integration with PBTSO Swarm Orchestrator.
    
    Enables multi-agent ARK operations:
    - Parallel distillation
    - Swarm commits
    - A2A coordination
    """
    
    def __init__(self):
        self.swarm_available = False
    
    def spawn_distill_swarm(self, sources: List[str], target: str) -> bool:
        """
        Spawn parallel distillation agents.
        
        Returns True if swarm was started.
        """
        # Future: integrate with tmux_swarm_orchestrator
        logger.info(f"PBTSO: Would spawn swarm for {len(sources)} sources")
        return False


# Factory function for full integration
def create_full_integration(repo_path: str) -> Dict:
    """Create all integration components."""
    return {
        "bus": ArkBusIntegration(Path(repo_path) / ".pluribus" / "bus" / "events.ndjson"),
        "ohm": OHMIntegration(),
        "laser": LASERIntegration(),
        "pbtso": PBTSOIntegration()
    }
