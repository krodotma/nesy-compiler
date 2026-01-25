"""
VIL (Vision-Integration-Learning) Interfaces
Abstract interfaces for vision-metalearning integration components.

These interfaces define the contracts that all VIL components must implement
for proper integration with the VILCoordinator.

Version: 1.0
Date: 2026-01-25
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


# === Geometric Types ===

@dataclass
class GeometricEmbedding:
    """Geometric embedding with manifold metadata."""
    vector: npt.NDArray[np.float64]
    manifold: str  # "spherical", "hyperbolic", "euclidean", "fiber_bundle"
    curvature: float = 1.0  # Gaussian curvature
    dimension: int = 0

    def __post_init__(self):
        if self.dimension == 0:
            self.dimension = len(self.vector)


@dataclass
class AttractorDynamics:
    """Attractor basin dynamics for energy-based systems."""
    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    energy: float
    basin_id: Optional[str] = None
    convergence_rate: float = 0.0


# === Vision Interface ===

class VILVisionInterface(ABC):
    """
    Interface for vision input components.

    Integrates:
    - VisionEye (WebRTC screen capture)
    - Theia capture module
    - MetaIngest inflow
    """

    @abstractmethod
    async def capture_frame(self) -> Optional[Dict[str, Any]]:
        """
        Capture a single frame from vision source.

        Returns:
            Frame dict with keys:
            - frame_id: str
            - image_data: str (base64)
            - width: int
            - height: int
            - timestamp: str
            - entropy: float (optional)
        """
        pass

    @abstractmethod
    async def get_ring_buffer(self) -> List[Dict[str, Any]]:
        """
        Get current ring buffer contents.

        Returns:
            List of frames in buffer (max size typically 60)
        """
        pass

    @abstractmethod
    async def analyze_entropy(self, frame: Dict[str, Any]) -> float:
        """
        Calculate visual entropy of frame.

        Uses H* (H-star) entropy normalization.
        """
        pass

    @abstractmethod
    async def emit_capture_event(self, frame: Dict[str, Any]) -> None:
        """Emit vision capture event to bus."""
        pass


# === Learning Interface ===

class VILLearningInterface(ABC):
    """
    Interface for learning and metalearning components.

    Integrates:
    - Omega Metalearning (L8 Gödel machines)
    - Learning Tower L8
    - Agent Lightning Trainer (RL)
    - Meta Learner (pattern recognition)
    """

    @abstractmethod
    async def add_task(
        self,
        task_id: str,
        prompt: str,
        response: str,
        success: bool = True,
    ) -> Dict[str, Any]:
        """
        Add a learning task for meta-learning.

        Used by Omega Metalearning for MAML-style adaptation.
        """
        pass

    @abstractmethod
    async def inner_loop(
        self,
        task_id: str,
        num_steps: int = 5,
        lr: float = 0.1,
    ) -> Dict[str, float]:
        """
        Run inner loop adaptation (fast, task-specific).

        MAML inner loop:
        θ'_i = θ_i - α ∇_θ L_Ti(θ_i)

        Returns:
            Dict with loss, accuracy, gradient_norm
        """
        pass

    @abstractmethod
    async def outer_loop(
        self,
        num_tasks: int = 10,
        meta_lr: float = 0.01,
    ) -> Dict[str, float]:
        """
        Run meta-learning outer loop (slow, across tasks).

        MAML outer loop:
        θ ← θ - β ∑_i ∇_θ L_Ti(θ'_i)

        Returns:
            Dict with meta_loss, meta_accuracy, task_diversity
        """
        pass

    @abstractmethod
    async def record_episode(
        self,
        episode_id: str,
        states: List[Any],
        actions: List[Any],
        rewards: List[float],
        next_states: List[Any],
    ) -> None:
        """
        Record RL episode for experience replay.

        Used by Agent Lightning Trainer.
        """
        pass

    @abstractmethod
    async def compute_meta_gradient(
        self,
        tasks: List[Dict[str, Any]],
    ) -> npt.NDArray[np.float64]:
        """
        Compute meta-gradient across multiple tasks.

        Gödel machine style: ∇_θ J(θ) where J is self-improvement.
        """
        pass


# === Synthesis Interface ===

class VILSynthesisInterface(ABC):
    """
    Interface for program synthesis components.

    Integrates:
    - CGP (Cartesian Genetic Programming)
    - EGGP (Graph-based program evolution)
    - VLM Specialist (program distillation)
    """

    @abstractmethod
    async def synthesize_program(
        self,
        source_image: str,
        target_goal: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize program from visual input.

        Returns:
            Dict with:
            - program_id: str
            - generated_code: str
            - confidence: float
            - genome_id: str (if CGP)
        """
        pass

    @abstractmethod
    async def mutate_genome(
        self,
        genome_id: str,
        mutation_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Apply mutation to CGP/EGGP genome.

        Mutation types:
        - point: Single node mutation
        - neutral: Neutral drift (preserves fitness)
        - structural: Add/remove nodes
        """
        pass

    @abstractmethod
    async def evaluate_fitness(
        self,
        program: str,
        test_cases: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate program fitness on test cases.

        Returns fitness score [0, 1].
        """
        pass

    @abstractmethod
    async def distill_from_teacher(
        self,
        teacher_outputs: List[Dict[str, Any]],
        num_examples: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Distill knowledge from teacher model.

        Used by VLM Specialist to learn from frontier models.
        """
        pass


# === Geometric Interface ===

class VILGeometricInterface(ABC):
    """
    Interface for geometric learning components.

    Integrates:
    - S^n (spherical) embeddings: Local geometry
    - H^n (hyperbolic) embeddings: Hierarchical structure
    - Fiber bundles: Parallel transport, curvature
    - Attractor dynamics: Energy landscapes
    """

    @abstractmethod
    async def embed_spherical(
        self,
        data: npt.NDArray[np.float64],
        dimension: int = 128,
    ) -> GeometricEmbedding:
        """
        Embed data in spherical manifold S^n.

        Spherical: ||x|| = 1, great circle distance
        Good for: Local neighborhoods, cyclical features
        """
        pass

    @abstractmethod
    async def embed_hyperbolic(
        self,
        data: npt.NDArray[np.float64],
        dimension: int = 128,
        curvature: float = -1.0,
    ) -> GeometricEmbedding:
        """
        Embed data in hyperbolic manifold H^n.

        Hyperbolic: Poincaré ball model
        Good for: Hierarchies, trees, exponential growth
        """
        pass

    @abstractmethod
    async def parallel_transport(
        self,
        vector: npt.NDArray[np.float64],
        from_point: npt.NDArray[np.float64],
        to_point: npt.NDArray[np.float64],
        manifold: str = "spherical",
    ) -> npt.NDArray[np.float64]:
        """
        Parallel transport vector between points on manifold.

        Used in fiber bundle geometry for curvature-aware updates.
        """
        pass

    @abstractmethod
    async def compute_curvature(
        self,
        point: npt.NDArray[np.float64],
        manifold: str = "spherical",
    ) -> float:
        """
        Compute Gaussian curvature at point.

        Positive: Sphere-like
        Zero: Euclidean
        Negative: Hyperbolic
        """
        pass

    @abstractmethod
    async def find_attractor(
        self,
        initial_state: npt.NDArray[np.float64],
        energy_fn: callable,
        num_steps: int = 100,
    ) -> AttractorDynamics:
        """
        Find attractor basin using energy function.

        Used in Modern Hopfield Continuum (L5 mHC).
        """
        pass

    @abstractmethod
    async def sinkhorn_crystallize(
        self,
        cost_matrix: npt.NDArray[np.float64],
        entropy_reg: float = 0.1,
        num_iterations: int = 100,
    ) -> npt.NDArray[np.float64]:
        """
        Sinkhorn iteration for optimal transport crystallization.

        Birkhoff polytope projection (L6).
        """
        pass


# === CMP Interface ===

class VILCMPInterface(ABC):
    """
    Interface for Clade Manager Protocol integration.

    Integrates:
    - Clade Manager CMP
    - Evolutionary fitness tracking
    - Lineage provenance
    """

    @abstractmethod
    async def calculate_fitness(
        self,
        clade_id: str,
        metrics: Dict[str, float],
    ) -> float:
        """
        Calculate clade fitness with phi-weighted metrics.

        Formula:
        fitness = task_completion * PHI +
                  test_coverage * 1.0 +
                  bug_rate * (1/PHI) +
                  review_velocity * (1/PHI^2) +
                  divergence_ratio * (1/PHI^3)
        """
        pass

    @abstractmethod
    async def speciate(
        self,
        parent_clade: str,
        pressure: float = 1.0,
        mutation_rate: float = 0.1,
    ) -> str:
        """
        Create new clade from parent.

        Returns new clade ID.
        """
        pass

    @abstractmethod
    async def recommend_merge(
        self,
        clade_ids: List[str],
    ) -> Optional[Tuple[str, str]]:
        """
        Recommend clades for merging based on fitness.

        Merge when fitness similarity > golden ratio threshold.

        Returns:
            Tuple of (clade_a, clade_b) to merge, or None
        """
        pass

    @abstractmethod
    async def track_lineage(
        self,
        clade_id: str,
        parent_id: str,
        event_type: str = "birth",
    ) -> Dict[str, Any]:
        """
        Track clade lineage for evolutionary provenance.

        Returns lineage metadata with depth, descendants, etc.
        """
        pass


# === ICL+ Interface ===

class VILICLInterface(ABC):
    """
    Interface for In-Context Learning Plus (ICL+).

    Integrates:
    - VLM Specialist ICL buffer
    - Example selection strategies
    - Geometric continuous updates
    """

    @abstractmethod
    async def add_example(
        self,
        image: str,
        prompt: str,
        response: str,
        success: bool = True,
        embedding: Optional[npt.NDArray[np.float64]] = None,
    ) -> str:
        """
        Add ICL example to buffer.

        Returns example ID.
        """
        pass

    @abstractmethod
    async def select_examples(
        self,
        query: str,
        k: int = 5,
        strategy: str = "nearest",
    ) -> List[Dict[str, Any]]:
        """
        Select k examples for in-context learning.

        Strategies:
        - nearest: Closest in embedding space
        - diverse: Maximize diversity
        - recent: Most recent
        - random: Random sampling
        """
        pass

    @abstractmethod
    async def update_geometrically(
        self,
        example: Dict[str, Any],
        manifold: str = "hyperbolic",
    ) -> None:
        """
        Update example embedding geometrically.

        Continuous learning on the manifold.
        """
        pass


# === Composite Interface ===

class VILInterface(
    VILVisionInterface,
    VILLearningInterface,
    VILSynthesisInterface,
    VILGeometricInterface,
    VILCMPInterface,
    VILICLInterface,
):
    """
    Composite interface combining all VIL interfaces.

    Components that implement all interfaces can inherit from this.
    """

    pass


__all__ = [
    # Geometric types
    "GeometricEmbedding",
    "AttractorDynamics",

    # Core interfaces
    "VILVisionInterface",
    "VILLearningInterface",
    "VILSynthesisInterface",
    "VILGeometricInterface",
    "VILCMPInterface",
    "VILICLInterface",

    # Composite
    "VILInterface",
]
