"""
Theia Temporal Sync — CTM-Derived Neural Synchronization for ARC Prescience.

ATTRIBUTION:
This module derives concepts from the Continuous Thought Machine (CTM):
    - Darlow, L., Regan, C., Risi, S., Seely, J., Jones, L.
    - "Continuous Thought Machines" (arXiv:2505.05522, May 2025)
    - SakanaAI: https://pub.sakana.ai/ctm/
    - License: Apache 2.0

CHERRY-PICKED CONCEPTS:
1. Neural Synchronization — encoding information in timing relationships
2. Internal Temporal Axis — decoupled time for reasoning
3. Per-Neuron History — unique temporal weights per unit
4. Hierarchical Temporal Representations — multi-scale patterns

APPLICATION TO ARC PRESCIENCE:
ARC (Abstraction & Reasoning Corpus) tasks require:
    - Pattern recognition at multiple scales
    - Temporal unfolding of abstract transformations
    - Synchronization between pattern detectors

This module adapts CTM's sync mechanism for:
    - Predicting next steps in abstract sequences
    - Detecting pattern phase-locks across scales
    - Enabling "prescience" — anticipating state transitions

Integration with Theia architecture:
    - L1 (Geometric): Sync over fiber bundle sections
    - L2 (mHC): Attractor synchronization
    - L3 (Birkhoff): Crystallization timing
    - L4 (DNA): State machine temporal coordination
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable
from enum import Enum, auto
import time


# =============================================================================
# CTM CORE CONCEPTS (DERIVED)
# =============================================================================

class SyncPhase(Enum):
    """
    Synchronization phases for temporal coordination.
    
    From CTM: Neural synchronization encodes information in timing.
    """
    DESYNC = auto()      # Neurons firing independently
    PARTIAL = auto()     # Some clusters synchronized
    FULL_SYNC = auto()   # Global synchronization (pattern lock)
    ANTIPHASE = auto()   # Alternating synchronization


@dataclass
class TemporalNeuron:
    """
    Single neuron with temporal history processing.
    
    CTM Concept: Each neuron uses unique weight parameters
    to process a history of incoming signals.
    
    Attributes:
        id: Unique neuron identifier
        history_length: How many timesteps to remember
        weights: Per-timestep weights for history processing
        activation_history: Ring buffer of past activations
        phase: Current oscillation phase (for sync detection)
    """
    id: str
    history_length: int = 16
    weights: np.ndarray = field(default_factory=lambda: np.zeros(16))
    activation_history: np.ndarray = field(default_factory=lambda: np.zeros(16))
    phase: float = 0.0
    
    def __post_init__(self):
        # Initialize with random temporal weights
        if np.all(self.weights == 0):
            self.weights = np.random.randn(self.history_length) * 0.1
        if len(self.activation_history) != self.history_length:
            self.activation_history = np.zeros(self.history_length)
    
    def step(self, input_signal: float) -> float:
        """
        Process input through temporal history.
        
        CTM: Unique per-neuron temporal processing.
        
        Args:
            input_signal: Current input
            
        Returns:
            Activation after temporal integration
        """
        # Shift history and add new input
        self.activation_history = np.roll(self.activation_history, 1)
        self.activation_history[0] = input_signal
        
        # Temporal convolution with learned weights
        temporal_response = np.dot(self.weights, self.activation_history)
        
        # Update phase based on activation pattern
        self.phase = np.arctan2(
            np.sin(self.phase) + 0.1 * temporal_response,
            np.cos(self.phase) + 0.1
        )
        
        # Nonlinear activation
        return np.tanh(temporal_response)


@dataclass
class SyncGroup:
    """
    Group of neurons that may synchronize.
    
    CTM Concept: Neural synchronisation employed as latent representation.
    """
    id: str
    neurons: List[TemporalNeuron] = field(default_factory=list)
    sync_threshold: float = 0.8  # Phase coherence threshold
    
    def compute_coherence(self) -> float:
        """
        Compute phase coherence (Kuramoto order parameter).
        
        R = |1/N Σ exp(i·θ_j)| where θ_j is phase of neuron j
        R = 1 → perfect sync, R = 0 → complete desync
        """
        if not self.neurons:
            return 0.0
        
        phases = np.array([n.phase for n in self.neurons])
        complex_phases = np.exp(1j * phases)
        order_param = np.abs(np.mean(complex_phases))
        
        return float(order_param)
    
    def get_sync_phase(self) -> SyncPhase:
        """Determine synchronization phase."""
        coherence = self.compute_coherence()
        
        if coherence > self.sync_threshold:
            return SyncPhase.FULL_SYNC
        elif coherence > 0.5:
            return SyncPhase.PARTIAL
        elif coherence < 0.2:
            # Check for antiphase
            phases = np.array([n.phase for n in self.neurons])
            if np.std(np.abs(np.diff(phases))) < 0.3:
                return SyncPhase.ANTIPHASE
        
        return SyncPhase.DESYNC
    
    def step_all(self, inputs: np.ndarray) -> np.ndarray:
        """Step all neurons with given inputs."""
        outputs = []
        for i, neuron in enumerate(self.neurons):
            inp = inputs[i] if i < len(inputs) else 0.0
            outputs.append(neuron.step(inp))
        return np.array(outputs)


# =============================================================================
# HIERARCHICAL TEMPORAL REPRESENTATION
# =============================================================================

@dataclass 
class TemporalHierarchy:
    """
    Multi-scale temporal representation.
    
    CTM Insight: Hierarchical temporal representations enable
    abstract reasoning by capturing patterns at different timescales.
    
    For ARC Prescience:
        - Fast scale: Pixel-level changes
        - Medium scale: Object transformations
        - Slow scale: Rule patterns
    """
    scales: List[SyncGroup] = field(default_factory=list)
    scale_factors: List[int] = field(default_factory=lambda: [1, 4, 16])
    
    def __post_init__(self):
        if not self.scales:
            # Create default hierarchy
            for i, factor in enumerate(self.scale_factors):
                neurons = [
                    TemporalNeuron(
                        id=f"L{i}_N{j}",
                        history_length=8 * factor,
                    )
                    for j in range(16 // factor)
                ]
                self.scales.append(SyncGroup(
                    id=f"scale_{factor}",
                    neurons=neurons,
                ))
    
    def step(self, input_signal: np.ndarray) -> Dict[str, Any]:
        """
        Step all scales with input.
        
        Returns sync state at each scale.
        """
        result = {
            "scales": [],
            "global_coherence": 0.0,
            "prescience_ready": False,
        }
        
        for scale in self.scales:
            # Downsample input for this scale
            scale_input = input_signal[:len(scale.neurons)]
            outputs = scale.step_all(scale_input)
            coherence = scale.compute_coherence()
            
            result["scales"].append({
                "id": scale.id,
                "coherence": coherence,
                "phase": scale.get_sync_phase().name,
                "outputs": outputs.tolist(),
            })
        
        # Global coherence across scales
        all_phases = []
        for scale in self.scales:
            all_phases.extend([n.phase for n in scale.neurons])
        
        if all_phases:
            complex_phases = np.exp(1j * np.array(all_phases))
            result["global_coherence"] = float(np.abs(np.mean(complex_phases)))
        
        # Prescience is ready when hierarchy is synchronized
        result["prescience_ready"] = result["global_coherence"] > 0.7
        
        return result


# =============================================================================
# ARC PRESCIENCE ENGINE
# =============================================================================

@dataclass
class PrescienceState:
    """
    State for ARC prescience predictions.
    
    Prescience = Anticipating next step based on synchronized patterns.
    """
    confidence: float = 0.0
    predicted_next: Optional[np.ndarray] = None
    pattern_id: Optional[str] = None
    reasoning_steps: List[str] = field(default_factory=list)


class ARCPrescienceEngine:
    """
    ARC Prescience Engine — CTM-derived temporal sync for abstract reasoning.
    
    Uses hierarchical temporal representations to:
    1. Detect patterns at multiple scales
    2. Synchronize pattern detectors
    3. Predict next steps (prescience)
    
    Integration Points:
        - Theia L2 (mHC): Pattern storage as attractors
        - Theia L3 (Birkhoff): Crystallization of predictions
        - Dashboard: Real-time sync visualization
    """
    
    def __init__(
        self,
        n_neurons: int = 64,
        n_scales: int = 3,
        sync_threshold: float = 0.75,
    ):
        self.n_neurons = n_neurons
        self.n_scales = n_scales
        self.sync_threshold = sync_threshold
        
        # Create temporal hierarchy
        scale_factors = [2**i for i in range(n_scales)]
        self.hierarchy = TemporalHierarchy(scale_factors=scale_factors)
        
        # Pattern memory (for ARC-style reasoning)
        self.pattern_memory: Dict[str, np.ndarray] = {}
        self.transition_memory: Dict[Tuple[str, str], float] = {}
        
        # Internal time (decoupled from input time)
        self.internal_time = 0
        self.time_step = 0.01
    
    def ingest(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Ingest observation and update temporal state.
        
        Args:
            observation: Current observation (e.g., ARC grid state)
            
        Returns:
            Sync status and prescience readiness
        """
        # Flatten observation for temporal neurons
        flat_obs = observation.flatten()
        
        # Update hierarchy
        result = self.hierarchy.step(flat_obs)
        
        # Advance internal time
        self.internal_time += self.time_step
        
        result["internal_time"] = self.internal_time
        
        return result
    
    def predict_next(
        self,
        current: np.ndarray,
        n_steps: int = 1,
    ) -> PrescienceState:
        """
        Predict next state(s) using temporal sync patterns.
        
        ARC Prescience: Anticipate transformations based on
        synchronized pattern representations.
        
        Args:
            current: Current state
            n_steps: How many steps to predict
            
        Returns:
            PrescienceState with prediction and confidence
        """
        state = PrescienceState()
        
        # Ingest current to update sync state
        sync_result = self.ingest(current)
        
        state.reasoning_steps.append(
            f"Global coherence: {sync_result['global_coherence']:.3f}"
        )
        
        if not sync_result["prescience_ready"]:
            state.reasoning_steps.append(
                "Hierarchy not synchronized - prescience uncertain"
            )
            state.confidence = sync_result["global_coherence"] * 0.5
            return state
        
        # Find matching pattern in memory
        current_hash = self._hash_pattern(current)
        
        if current_hash in self.pattern_memory:
            state.pattern_id = current_hash
            state.reasoning_steps.append(f"Matched pattern: {current_hash}")
            
            # Look for learned transitions
            for (src, dst), prob in self.transition_memory.items():
                if src == current_hash:
                    if dst in self.pattern_memory:
                        state.predicted_next = self.pattern_memory[dst]
                        state.confidence = prob * sync_result["global_coherence"]
                        state.reasoning_steps.append(
                            f"Transition {src}→{dst} with p={prob:.2f}"
                        )
                        break
        else:
            state.reasoning_steps.append("Novel pattern - storing for future")
            self.pattern_memory[current_hash] = current.copy()
        
        return state
    
    def learn_transition(
        self,
        before: np.ndarray,
        after: np.ndarray,
        success: bool = True,
    ) -> None:
        """
        Learn pattern transition from observation.
        
        Builds transition memory for prescience.
        """
        before_hash = self._hash_pattern(before)
        after_hash = self._hash_pattern(after)
        
        # Store patterns
        self.pattern_memory[before_hash] = before.copy()
        self.pattern_memory[after_hash] = after.copy()
        
        # Update transition probability
        key = (before_hash, after_hash)
        current_prob = self.transition_memory.get(key, 0.5)
        
        if success:
            self.transition_memory[key] = min(1.0, current_prob + 0.1)
        else:
            self.transition_memory[key] = max(0.0, current_prob - 0.1)
    
    def _hash_pattern(self, pattern: np.ndarray) -> str:
        """Hash pattern for memory lookup."""
        import hashlib
        return hashlib.md5(pattern.tobytes()).hexdigest()[:16]
    
    def get_sync_visualization(self) -> Dict[str, Any]:
        """
        Get data for dashboard sync visualization.
        
        For integration with Pluribus dashboard.
        """
        viz = {
            "type": "temporal_sync",
            "internal_time": self.internal_time,
            "scales": [],
        }
        
        for scale in self.hierarchy.scales:
            scale_viz = {
                "id": scale.id,
                "n_neurons": len(scale.neurons),
                "coherence": scale.compute_coherence(),
                "phase": scale.get_sync_phase().name,
                "neuron_phases": [n.phase for n in scale.neurons],
            }
            viz["scales"].append(scale_viz)
        
        return viz


# =============================================================================
# UI/UX INTEGRATION HELPERS
# =============================================================================

def create_sync_meter_data(engine: ARCPrescienceEngine) -> Dict[str, Any]:
    """
    Create data for a sync meter UI component.
    
    For dashboard integration.
    """
    viz = engine.get_sync_visualization()
    
    return {
        "coherence": viz["scales"][0]["coherence"] if viz["scales"] else 0,
        "phase": viz["scales"][0]["phase"] if viz["scales"] else "DESYNC",
        "internal_time": viz["internal_time"],
        "multi_scale": [s["coherence"] for s in viz["scales"]],
        "ready": all(s["coherence"] > 0.7 for s in viz["scales"]),
    }


def create_hierarchy_viz(engine: ARCPrescienceEngine) -> List[Dict[str, Any]]:
    """
    Create hierarchical visualization data.
    
    For dashboard TreeMap or nested display.
    """
    viz = engine.get_sync_visualization()
    
    nodes = []
    for scale in viz["scales"]:
        node = {
            "name": scale["id"],
            "coherence": scale["coherence"],
            "phase": scale["phase"],
            "children": [
                {"name": f"N{i}", "phase": p}
                for i, p in enumerate(scale["neuron_phases"])
            ],
        }
        nodes.append(node)
    
    return nodes


__all__ = [
    # Core Types
    "SyncPhase",
    "TemporalNeuron",
    "SyncGroup",
    "TemporalHierarchy",
    # Prescience
    "PrescienceState",
    "ARCPrescienceEngine",
    # UI/UX Helpers
    "create_sync_meter_data",
    "create_hierarchy_viz",
]
