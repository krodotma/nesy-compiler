"""
Theia Reentry — Energy landscape modification via coalgebraic reentry.

The reentry mechanism allows higher-level states to modify
the dynamics of lower-level computation:

    Q → F(Q) × Mod(E)

where:
    - Q is the state space (DNA automaton)
    - F(Q) is the next-state functor
    - Mod(E) is the energy modification space

This enables self-teaching: the system can improve its own
energy landscapes based on execution experience.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import time


@dataclass
class EnergyModification:
    """
    Specification for modifying an energy landscape.
    
    Attributes:
        target: Which energy landscape to modify
        operation: Type of modification
        parameters: Modification parameters
        priority: Higher priority mods applied first
    """
    target: str  # "mhc", "birkhoff", "temporal"
    operation: str  # "scale_beta", "shift_attractor", "add_pattern", "remove_pattern"
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "operation": self.operation,
            "parameters": self.parameters,
            "priority": self.priority,
            "timestamp": self.timestamp,
        }


class ModificationFunctor:
    """
    The Mod(E) functor — transforms energy landscapes.
    
    Collects modifications from DNA automaton execution
    and applies them to lower layers.
    """
    
    def __init__(self):
        self._pending: List[EnergyModification] = []
        self._applied: List[EnergyModification] = []
    
    def queue(self, mod: EnergyModification) -> None:
        """Queue a modification for later application."""
        self._pending.append(mod)
        self._pending.sort(key=lambda m: -m.priority)  # Higher priority first
    
    def peek(self) -> Optional[EnergyModification]:
        """Peek at next modification without removing it."""
        return self._pending[0] if self._pending else None
    
    def pop(self) -> Optional[EnergyModification]:
        """Pop and return next modification."""
        if self._pending:
            mod = self._pending.pop(0)
            self._applied.append(mod)
            return mod
        return None
    
    def apply_to_mhc(self, memory, mod: EnergyModification) -> bool:
        """
        Apply modification to mHC memory.
        
        Operations:
            - scale_beta: Multiply beta by factor
            - shift_attractor: Move attractor position
            - add_pattern: Add new pattern
            - remove_pattern: Remove pattern by index
        """
        if mod.target != "mhc":
            return False
        
        try:
            if mod.operation == "scale_beta":
                factor = mod.parameters.get("factor", 1.0)
                memory.beta *= factor
                return True
            
            elif mod.operation == "add_pattern":
                pattern = mod.parameters.get("pattern")
                if pattern is not None:
                    memory.add_pattern(np.array(pattern))
                    return True
            
            elif mod.operation == "remove_pattern":
                idx = mod.parameters.get("index", -1)
                if 0 <= idx < len(memory.patterns):
                    memory.patterns = np.delete(memory.patterns, idx, axis=0)
                    return True
            
            elif mod.operation == "shift_attractor":
                idx = mod.parameters.get("index", 0)
                delta = mod.parameters.get("delta", 0.0)
                if 0 <= idx < len(memory.patterns):
                    memory.patterns[idx] += delta
                    return True
        except Exception:
            pass
        
        return False
    
    def apply_to_birkhoff(self, sinkhorn_params: Dict[str, float], mod: EnergyModification) -> Dict[str, float]:
        """
        Apply modification to Birkhoff crystallization parameters.
        
        Operations:
            - scale_pressure: Multiply crystallization pressure
            - adjust_lambda: Modify Lyapunov coupling
        """
        if mod.target != "birkhoff":
            return sinkhorn_params
        
        result = sinkhorn_params.copy()
        
        if mod.operation == "scale_pressure":
            factor = mod.parameters.get("factor", 1.0)
            result["pressure"] = result.get("pressure", 1.0) * factor
        
        elif mod.operation == "adjust_lambda":
            delta = mod.parameters.get("delta", 0.0)
            result["lambda"] = result.get("lambda", 0.5) + delta
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get modification statistics."""
        return {
            "pending": len(self._pending),
            "applied": len(self._applied),
            "by_target": {
                "mhc": sum(1 for m in self._applied if m.target == "mhc"),
                "birkhoff": sum(1 for m in self._applied if m.target == "birkhoff"),
                "temporal": sum(1 for m in self._applied if m.target == "temporal"),
            },
        }


class ReentryController:
    """
    Controls reentry from DNA automaton to lower layers.
    
    Monitors automaton execution and generates appropriate
    energy modifications based on outcomes.
    """
    
    def __init__(self):
        self.functor = ModificationFunctor()
        self._success_count = 0
        self._failure_count = 0
        self._learning_rate = 0.1
    
    def on_success(self, context: Dict[str, Any]) -> None:
        """
        Handle successful automaton execution.
        
        Successful execution → reinforce current energy configuration.
        """
        self._success_count += 1
        
        # Reinforce by slightly increasing beta (sharper memory)
        self.functor.queue(EnergyModification(
            target="mhc",
            operation="scale_beta",
            parameters={"factor": 1.0 + self._learning_rate * 0.1},
            priority=1,
        ))
        
        # Record success pattern if available
        if "pattern" in context:
            self.functor.queue(EnergyModification(
                target="mhc",
                operation="add_pattern",
                parameters={"pattern": context["pattern"]},
                priority=2,
            ))
    
    def on_failure(self, context: Dict[str, Any]) -> None:
        """
        Handle failed automaton execution.
        
        Failed execution → explore by softening energy landscape.
        """
        self._failure_count += 1
        
        # Soften landscape by decreasing beta (broader search)
        self.functor.queue(EnergyModification(
            target="mhc",
            operation="scale_beta",
            parameters={"factor": 1.0 - self._learning_rate * 0.1},
            priority=1,
        ))
        
        # Increase crystallization pressure to force convergence
        self.functor.queue(EnergyModification(
            target="birkhoff",
            operation="scale_pressure",
            parameters={"factor": 1.0 + self._learning_rate * 0.2},
            priority=1,
        ))
    
    def apply_pending(self, memory=None, birkhoff_params: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Apply all pending modifications to components.
        
        Returns summary of applied modifications.
        """
        results = {
            "mhc_applied": 0,
            "birkhoff_applied": 0,
            "new_params": birkhoff_params.copy() if birkhoff_params else {},
        }
        
        while True:
            mod = self.functor.pop()
            if mod is None:
                break
            
            if mod.target == "mhc" and memory is not None:
                if self.functor.apply_to_mhc(memory, mod):
                    results["mhc_applied"] += 1
            
            elif mod.target == "birkhoff" and birkhoff_params is not None:
                results["new_params"] = self.functor.apply_to_birkhoff(
                    results["new_params"], mod
                )
                results["birkhoff_applied"] += 1
        
        return results
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get reentry learning statistics."""
        total = self._success_count + self._failure_count
        return {
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "success_rate": self._success_count / max(1, total),
            "learning_rate": self._learning_rate,
            "functor_stats": self.functor.get_stats(),
        }


# =============================================================================
# SELF-TEACHING INTERFACE
# =============================================================================

class SelfTeacher:
    """
    Self-teaching via reentry.
    
    Observes execution, generates modifications,
    and applies them to improve future performance.
    """
    
    def __init__(self):
        self.controller = ReentryController()
        self._episodes: List[Dict[str, Any]] = []
    
    def record_episode(
        self,
        observations: List[Any],
        actions: List[str],
        success: bool,
    ) -> None:
        """Record an execution episode for learning."""
        episode = {
            "observations": observations,
            "actions": actions,
            "success": success,
            "timestamp": time.time(),
        }
        self._episodes.append(episode)
        
        # Trigger reentry based on outcome
        if success:
            self.controller.on_success({"pattern": observations[-1] if observations else None})
        else:
            self.controller.on_failure({})
    
    def teach(self, memory=None, birkhoff_params: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Apply learned modifications from recent episodes.
        
        Returns teaching summary.
        """
        if not self._episodes:
            return {"status": "no_episodes"}
        
        results = self.controller.apply_pending(memory, birkhoff_params)
        
        results["episodes_processed"] = len(self._episodes)
        results["learning_stats"] = self.controller.get_learning_stats()
        
        return results
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard visualization."""
        stats = self.controller.get_learning_stats()
        return {
            "episodes": len(self._episodes),
            "success_rate": stats["success_rate"],
            "pending_mods": stats["functor_stats"]["pending"],
            "applied_mods": stats["functor_stats"]["applied"],
        }


__all__ = [
    "EnergyModification",
    "ModificationFunctor",
    "ReentryController",
    "SelfTeacher",
]
