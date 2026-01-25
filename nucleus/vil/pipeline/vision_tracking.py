"""
VIL Vision Tracking (CMP + Lineage)
Tracks vision evolution through CMP metrics and lineage provenance.

Based on:
- Clade Manager Protocol (CMP) for fitness tracking
- MetaIngest Lineageflow for evolutionary provenance

Features:
1. Visual capture CMP metrics (quality, analysis, lineage)
2. Vision lineage tracking with provenance
3. CMP score calculation with phi-weighting
4. Generation tracking for vision evolution
5. Lineage tree visualization

Version: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class VisionState(str, Enum):
    """Vision processing states."""

    CAPTURED = "captured"
    ANALYZED = "analyzed"
    ACCEPTED = "accepted"  # Passed quality filter
    REJECTED = "rejected"  # Failed quality filter
    ICL_ADDED = "icl_added"  # Added to ICL buffer


@dataclass
class VisionLineage:
    """
    Lineage provenance for vision frames.

    Tracks:
    - Parent frames (for evolutionary tracking)
    - Generation number
    - Branch depth in lineage tree
    - Ancestry chain
    """

    lineage_id: str
    parent_id: Optional[str] = None
    generation: int = 0
    branch_depth: int = 0
    ancestry: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    def add_child(self, child_id: str) -> None:
        """Add child to lineage."""
        self.children.append(child_id)

    def get_ancestry_string(self) -> str:
        """Get ancestry as string representation."""
        if not self.ancestry:
            return f"gen{self.generation}"
        return " → ".join([f"gen{i}" for i in self.ancestry[-3:]] + [f"gen{self.generation}"])


@dataclass
class VisionCMPMetrics:
    """
    CMP (Code Maturity Potential) metrics for vision frames.

    Based on Clade Manager phi-weighted fitness:
    fitness = task_completion * PHI +
              test_coverage * 1.0 +
              bug_rate * (1/PHI) +
              review_velocity * (1/PHI^2) +
              divergence_ratio * (1/PHI^3)
    """

    # Vision-specific metrics
    capture_quality: float = 0.0  # Frame buffer quality (H* based)
    analysis_confidence: float = 0.0  # VLM inference confidence
    icl_value: float = 0.0  # Value for ICL (novelty)

    # CMP calculation
    phi_weighted_fitness: float = 0.0
    generation: int = 0

    # Composite score
    composite_score: float = 0.0  # Overall quality [0, 1]

    def calculate_fitness(self, phi: float = 1.618) -> float:
        """
        Calculate phi-weighted fitness score.

        Formula:
        fitness = capture_quality * PHI +
                  analysis_confidence * 1.0 +
                  icl_value * (1/PHI)
        """
        self.phi_weighted_fitness = (
            self.capture_quality * phi +
            self.analysis_confidence * 1.0 +
            self.icl_value * (1 / phi)
        ) / (phi + 1.0 + 1/phi)  # Normalize

        # Composite: weighted average
        self.composite_score = (
            self.capture_quality * 0.4 +
            self.analysis_confidence * 0.4 +
            self.icl_value * 0.2
        )

        return self.phi_weighted_fitness


@dataclass
class VisionFrame:
    """
    Single vision frame with full tracking.

    Combines:
    - Frame data
    - CMP metrics
    - Lineage provenance
    - Processing state
    """

    frame_id: str
    image_data: str  # Base64 encoded
    timestamp: float

    # Metrics
    cmp: VisionCMPMetrics
    lineage: VisionLineage

    # State
    state: VisionState = VisionState.CAPTURED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "state": self.state.value,
            "cmp": {
                "capture_quality": self.cmp.capture_quality,
                "analysis_confidence": self.cmp.analysis_confidence,
                "icl_value": self.cmp.icl_value,
                "phi_weighted_fitness": self.cmp.phi_weighted_fitness,
                "composite_score": self.cmp.composite_score,
                "generation": self.cmp.generation,
            },
            "lineage": {
                "lineage_id": self.lineage.lineage_id,
                "parent_id": self.lineage.parent_id,
                "generation": self.lineage.generation,
                "branch_depth": self.lineage.branch_depth,
                "ancestry_string": self.lineage.get_ancestry_string(),
                "num_children": len(self.lineage.children),
            },
            "image_data": self.image_data if include_image else None,
            "metadata": self.metadata,
        }


class VisionTracker:
    """
    Tracks vision frames with CMP and lineage.

    Features:
    1. CMP calculation with phi-weighting
    2. Lineage tree management
    3. Generation tracking
    4. State transitions
    5. Bus event emission
    """

    def __init__(
        self,
        phi: float = 1.618033988749895,
        bus_emitter: Optional[callable] = None,
    ):
        self.phi = phi
        self.bus_emitter = bus_emitter

        # Storage
        self.frames: Dict[str, VisionFrame] = {}
        self.lineages: Dict[str, VisionLineage] = {}

        # Statistics
        self.stats = {
            "total_frames": 0,
            "by_state": {s.value: 0 for s in VisionState},
            "avg_composite_score": 0.0,
            "max_generation": 0,
            "total_lineages": 0,
        }

    def create_frame(
        self,
        image_data: str,
        capture_quality: float = 0.0,
        analysis_confidence: float = 0.0,
        icl_value: float = 0.0,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VisionFrame:
        """
        Create new vision frame with tracking.

        Args:
            image_data: Base64-encoded image
            capture_quality: H* based quality [0, 1]
            analysis_confidence: VLM confidence [0, 1]
            icl_value: ICL novelty value [0, 1]
            parent_id: Parent frame ID for lineage
            metadata: Additional metadata

        Returns:
            VisionFrame with CMP and lineage
        """
        frame_id = f"frame_{int(time.time() * 1000)}_{self.stats['total_frames']}"

        # Create CMP metrics
        cmp = VisionCMPMetrics(
            capture_quality=capture_quality,
            analysis_confidence=analysis_confidence,
            icl_value=icl_value,
        )
        cmp.calculate_fitness(self.phi)

        # Create lineage
        if parent_id and parent_id in self.frames:
            parent_frame = self.frames[parent_id]
            parent_lineage = self.lineages[parent_frame.lineage.lineage_id]

            lineage_id = parent_lineage.lineage_id
            generation = parent_lineage.generation + 1
            branch_depth = parent_lineage.branch_depth + 1
            ancestry = parent_lineage.ancestry + [parent_lineage.generation]

            # Update parent
            parent_lineage.add_child(frame_id)
        else:
            # New lineage
            lineage_id = f"lineage_{len(self.lineages)}"
            generation = 0
            branch_depth = 0
            ancestry = []

            self.stats["total_lineages"] += 1

        lineage = VisionLineage(
            lineage_id=lineage_id,
            parent_id=parent_id,
            generation=generation,
            branch_depth=branch_depth,
            ancestry=ancestry,
        )

        # Create frame
        frame = VisionFrame(
            frame_id=frame_id,
            image_data=image_data,
            timestamp=time.time(),
            cmp=cmp,
            lineage=lineage,
            metadata=metadata or {},
        )

        # Store
        self.frames[frame_id] = frame
        if lineage_id not in self.lineages:
            self.lineages[lineage_id] = lineage

        # Update stats
        self.stats["total_frames"] += 1
        self.stats["by_state"][frame.state.value] += 1
        self.stats["max_generation"] = max(self.stats["max_generation"], generation)

        # Emit event
        self._emit_frame_event(frame, "created")

        return frame

    def update_state(
        self,
        frame_id: str,
        new_state: VisionState,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[VisionFrame]:
        """
        Update frame state and metrics.

        Args:
            frame_id: Frame to update
            new_state: New state
            additional_metrics: Optional additional CMP metrics

        Returns:
            Updated frame if found, None otherwise
        """
        if frame_id not in self.frames:
            return None

        frame = self.frames[frame_id]

        # Update state
        old_state = frame.state
        frame.state = new_state

        # Update metrics
        if additional_metrics:
            if "capture_quality" in additional_metrics:
                frame.cmp.capture_quality = additional_metrics["capture_quality"]
            if "analysis_confidence" in additional_metrics:
                frame.cmp.analysis_confidence = additional_metrics["analysis_confidence"]
            if "icl_value" in additional_metrics:
                frame.cmp.icl_value = additional_metrics["icl_value"]

            # Recalculate fitness
            frame.cmp.calculate_fitness(self.phi)

        # Update stats
        self.stats["by_state"][old_state.value] -= 1
        self.stats["by_state"][new_state.value] += 1

        # Update average score
        total_score = sum(f.cmp.composite_score for f in self.frames.values())
        self.stats["avg_composite_score"] = total_score / len(self.frames)

        # Emit event
        self._emit_frame_event(frame, "state_changed")

        return frame

    def get_lineage_tree(
        self,
        lineage_id: Optional[str] = None,
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """
        Get lineage tree for visualization.

        Args:
            lineage_id: Specific lineage or all
            max_depth: Maximum depth to traverse

        Returns:
            Tree structure for visualization
        """
        if lineage_id:
            lineages = {lineage_id: self.lineages.get(lineage_id)}
        else:
            lineages = self.lineages

        trees = {}
        for lid, lineage in lineages.items():
            if lineage is None:
                continue

            trees[lid] = {
                "lineage_id": lid,
                "generation": lineage.generation,
                "branch_depth": lineage.branch_depth,
                "ancestry_string": lineage.get_ancestry_string(),
                "num_children": len(lineage.children),
                "frames": [
                    {
                        "frame_id": fid,
                        "state": self.frames[fid].state.value,
                        "composite_score": self.frames[fid].cmp.composite_score,
                    }
                    for fid in self._get_lineage_frames(lid, max_depth)
                ],
            }

        return trees

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            **self.stats,
            "num_lineages": len(self.lineages),
            "avg_generation": sum(l.generation for l in self.lineages.values()) / max(1, len(self.lineages)),
            "avg_branch_depth": sum(l.branch_depth for l in self.lineages.values()) / max(1, len(self.lineages)),
        }

    # === Private Methods ===

    def _get_lineage_frames(self, lineage_id: str, max_depth: int) -> List[str]:
        """Get all frame IDs in lineage up to max_depth."""
        frames = []
        for frame_id, frame in self.frames.items():
            if frame.lineage.lineage_id != lineage_id:
                continue
            if frame.lineage.generation > max_depth:
                continue
            frames.append(frame_id)
        return frames

    def _emit_frame_event(self, frame: VisionFrame, event_type: str) -> None:
        """Emit frame tracking event to bus."""
        if not self.bus_emitter:
            return

        event = {
            "topic": f"vil.vision.{event_type}",
            "data": {
                "frame_id": frame.frame_id,
                "timestamp": time.time(),
                "state": frame.state.value,
                "cmp": frame.cmp.__dict__,
                "lineage": {
                    "lineage_id": frame.lineage.lineage_id,
                    "generation": frame.lineage.generation,
                    "ancestry_string": frame.lineage.get_ancestry_string(),
                },
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[VisionTracker] Bus emission error: {e}")


def create_vision_tracker(
    phi: float = 1.618033988749895,
    bus_emitter: Optional[callable] = None,
) -> VisionTracker:
    """Create vision tracker with default config."""
    return VisionTracker(phi=phi, bus_emitter=bus_emitter)


__all__ = [
    "VisionState",
    "VisionLineage",
    "VisionCMPMetrics",
    "VisionFrame",
    "VisionTracker",
    "create_vision_tracker",
]
