"""
Temporal Consistency Module
===========================

Maintains visual consistency across video frames.
Uses optical flow and feature matching to ensure smooth transitions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import math


class FlowMethod(Enum):
    """Optical flow computation methods."""
    LUCAS_KANADE = auto()      # Lucas-Kanade sparse flow
    FARNEBACK = auto()         # Farneback dense flow
    RAFT = auto()              # RAFT deep learning flow
    FLOWNET = auto()           # FlowNet
    PWC_NET = auto()           # PWC-Net
    GMF = auto()               # Global Motion Flow


class ConsistencyMode(Enum):
    """Modes for temporal consistency enforcement."""
    NONE = auto()              # No consistency enforcement
    SOFT = auto()              # Soft blending between frames
    HARD = auto()              # Strong consistency enforcement
    ADAPTIVE = auto()          # Adaptive based on motion


@dataclass
class MotionVector:
    """Represents motion between two points."""
    x: float
    y: float
    magnitude: float = field(init=False)
    angle: float = field(init=False)

    def __post_init__(self):
        self.magnitude = math.sqrt(self.x ** 2 + self.y ** 2)
        self.angle = math.atan2(self.y, self.x)

    def to_dict(self) -> Dict[str, float]:
        return {
            'x': self.x,
            'y': self.y,
            'magnitude': self.magnitude,
            'angle': self.angle,
        }


@dataclass
class FlowField:
    """
    Optical flow field between two frames.

    Attributes:
        width: Width of the flow field
        height: Height of the flow field
        vectors: Motion vectors (simplified representation)
        method: Flow computation method used
        confidence: Overall confidence in flow estimation
    """
    width: int
    height: int
    vectors: List[List[MotionVector]] = field(default_factory=list)
    method: FlowMethod = FlowMethod.FARNEBACK
    confidence: float = 1.0

    def get_average_motion(self) -> MotionVector:
        """Calculate average motion across the field."""
        if not self.vectors or not self.vectors[0]:
            return MotionVector(0.0, 0.0)

        total_x = 0.0
        total_y = 0.0
        count = 0

        for row in self.vectors:
            for vec in row:
                total_x += vec.x
                total_y += vec.y
                count += 1

        if count == 0:
            return MotionVector(0.0, 0.0)

        return MotionVector(total_x / count, total_y / count)

    def get_motion_at(self, x: int, y: int) -> Optional[MotionVector]:
        """Get motion vector at specific position."""
        if 0 <= y < len(self.vectors) and 0 <= x < len(self.vectors[y]):
            return self.vectors[y][x]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'width': self.width,
            'height': self.height,
            'method': self.method.name,
            'confidence': self.confidence,
            'average_motion': self.get_average_motion().to_dict(),
        }


@dataclass
class ConsistencyReport:
    """
    Report on temporal consistency between frames.

    Attributes:
        frame_index: Index of the frame analyzed
        consistency_score: Score from 0.0 (inconsistent) to 1.0 (consistent)
        motion_score: Motion coherence score
        color_score: Color consistency score
        artifact_detected: Whether visual artifacts were detected
        recommendations: Suggested fixes for inconsistencies
    """
    frame_index: int
    consistency_score: float
    motion_score: float = 1.0
    color_score: float = 1.0
    artifact_detected: bool = False
    recommendations: List[str] = field(default_factory=list)

    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if consistency is above threshold."""
        return self.consistency_score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_index': self.frame_index,
            'consistency_score': self.consistency_score,
            'motion_score': self.motion_score,
            'color_score': self.color_score,
            'artifact_detected': self.artifact_detected,
            'recommendations': self.recommendations,
        }


class TemporalConsistencyEngine:
    """
    Maintains temporal consistency across video frames.

    Uses optical flow and feature matching to:
    - Detect inconsistencies between frames
    - Blend frames for smooth transitions
    - Propagate style/content across time
    """

    def __init__(self,
                 flow_method: FlowMethod = FlowMethod.FARNEBACK,
                 consistency_mode: ConsistencyMode = ConsistencyMode.ADAPTIVE,
                 blend_strength: float = 0.5,
                 motion_threshold: float = 10.0):
        """
        Initialize the temporal consistency engine.

        Args:
            flow_method: Method for optical flow computation
            consistency_mode: Mode for enforcing consistency
            blend_strength: Strength of inter-frame blending (0.0-1.0)
            motion_threshold: Threshold for motion detection
        """
        self.flow_method = flow_method
        self.consistency_mode = consistency_mode
        self.blend_strength = blend_strength
        self.motion_threshold = motion_threshold

        self._frame_cache: Dict[int, Any] = {}
        self._flow_cache: Dict[Tuple[int, int], FlowField] = {}
        self._feature_cache: Dict[int, Any] = {}

    def compute_flow(self, frame1_path: Path, frame2_path: Path) -> FlowField:
        """
        Compute optical flow between two frames.

        Args:
            frame1_path: Path to first frame
            frame2_path: Path to second frame

        Returns:
            FlowField representing motion between frames.
        """
        # Simplified implementation - would use OpenCV/deep learning in production
        # For now, return a placeholder flow field

        width, height = 64, 36  # Downsampled for efficiency
        vectors = []

        for y in range(height):
            row = []
            for x in range(width):
                # Placeholder: assume small random motion
                vec = MotionVector(0.0, 0.0)
                row.append(vec)
            vectors.append(row)

        return FlowField(
            width=width,
            height=height,
            vectors=vectors,
            method=self.flow_method,
            confidence=0.9
        )

    def analyze_consistency(self,
                            frames: List[Path],
                            start_index: int = 0) -> List[ConsistencyReport]:
        """
        Analyze temporal consistency across a sequence of frames.

        Args:
            frames: List of paths to frame files
            start_index: Starting index for frame numbering

        Returns:
            List of ConsistencyReport for each frame pair.
        """
        reports = []

        for i in range(len(frames) - 1):
            frame_a = frames[i]
            frame_b = frames[i + 1]

            # Compute flow between frames
            flow = self.compute_flow(frame_a, frame_b)

            # Analyze motion coherence
            avg_motion = flow.get_average_motion()
            motion_score = self._compute_motion_score(flow, avg_motion)

            # Analyze color consistency (placeholder)
            color_score = self._compute_color_score(frame_a, frame_b)

            # Detect artifacts (placeholder)
            artifacts = self._detect_artifacts(frame_a, frame_b, flow)

            # Calculate overall consistency
            consistency_score = (motion_score * 0.4 + color_score * 0.4 +
                               (0.0 if artifacts else 0.2))

            # Generate recommendations
            recommendations = self._generate_recommendations(
                motion_score, color_score, artifacts, avg_motion
            )

            reports.append(ConsistencyReport(
                frame_index=start_index + i,
                consistency_score=consistency_score,
                motion_score=motion_score,
                color_score=color_score,
                artifact_detected=artifacts,
                recommendations=recommendations
            ))

        return reports

    def enforce_consistency(self,
                           frames: List[Path],
                           output_dir: Path,
                           reports: Optional[List[ConsistencyReport]] = None) -> List[Path]:
        """
        Enforce temporal consistency on a frame sequence.

        Args:
            frames: List of paths to frame files
            output_dir: Directory for output frames
            reports: Optional pre-computed consistency reports

        Returns:
            List of paths to consistency-enforced frames.
        """
        if self.consistency_mode == ConsistencyMode.NONE:
            return frames

        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []

        # Analyze if reports not provided
        if reports is None:
            reports = self.analyze_consistency(frames)

        for i, frame_path in enumerate(frames):
            output_path = output_dir / f"consistent_{i:06d}{frame_path.suffix}"

            if i == 0:
                # First frame - just copy
                self._copy_frame(frame_path, output_path)
            else:
                # Apply consistency based on report
                report = reports[i - 1] if i - 1 < len(reports) else None

                if report and not report.is_acceptable():
                    # Frame needs consistency enforcement
                    self._blend_frames(
                        frames[i - 1], frame_path, output_path,
                        self._get_blend_strength(report)
                    )
                else:
                    # Frame is acceptable
                    self._copy_frame(frame_path, output_path)

            output_paths.append(output_path)

        return output_paths

    def propagate_style(self,
                        reference_frame: Path,
                        target_frames: List[Path],
                        output_dir: Path,
                        style_strength: float = 0.7) -> List[Path]:
        """
        Propagate visual style from reference frame to targets.

        Args:
            reference_frame: Path to the style reference frame
            target_frames: Frames to apply style to
            output_dir: Directory for output frames
            style_strength: How strongly to apply style (0.0-1.0)

        Returns:
            List of paths to style-propagated frames.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []

        # Extract style features from reference (placeholder)
        style_features = self._extract_style_features(reference_frame)

        for i, target_path in enumerate(target_frames):
            output_path = output_dir / f"styled_{i:06d}{target_path.suffix}"

            # Apply style (placeholder implementation)
            self._apply_style(target_path, style_features, output_path, style_strength)
            output_paths.append(output_path)

        return output_paths

    def _compute_motion_score(self, flow: FlowField, avg_motion: MotionVector) -> float:
        """Compute motion coherence score."""
        if not flow.vectors:
            return 1.0

        # Check variance from average motion
        variance = 0.0
        count = 0

        for row in flow.vectors:
            for vec in row:
                diff_x = vec.x - avg_motion.x
                diff_y = vec.y - avg_motion.y
                variance += diff_x ** 2 + diff_y ** 2
                count += 1

        if count == 0:
            return 1.0

        avg_variance = variance / count

        # Convert variance to score (lower variance = higher score)
        score = 1.0 / (1.0 + avg_variance / 100.0)
        return min(1.0, max(0.0, score))

    def _compute_color_score(self, frame1: Path, frame2: Path) -> float:
        """Compute color consistency score between frames."""
        # Placeholder - would analyze color histograms
        return 0.9

    def _detect_artifacts(self, frame1: Path, frame2: Path, flow: FlowField) -> bool:
        """Detect visual artifacts between frames."""
        # Placeholder - would detect flickering, popping, etc.
        avg_motion = flow.get_average_motion()
        return avg_motion.magnitude > self.motion_threshold * 2

    def _generate_recommendations(self,
                                  motion_score: float,
                                  color_score: float,
                                  artifacts: bool,
                                  motion: MotionVector) -> List[str]:
        """Generate recommendations for fixing inconsistencies."""
        recommendations = []

        if motion_score < 0.5:
            recommendations.append("High motion variance detected. Consider regenerating with motion constraints.")

        if color_score < 0.7:
            recommendations.append("Color inconsistency detected. Apply color grading or histogram matching.")

        if artifacts:
            recommendations.append("Visual artifacts detected. Increase blending or reduce motion.")

        if motion.magnitude > self.motion_threshold:
            recommendations.append(f"High motion ({motion.magnitude:.1f}px). Consider intermediate frames.")

        return recommendations

    def _get_blend_strength(self, report: ConsistencyReport) -> float:
        """Calculate blend strength based on consistency report."""
        if self.consistency_mode == ConsistencyMode.SOFT:
            return self.blend_strength * 0.5
        elif self.consistency_mode == ConsistencyMode.HARD:
            return min(1.0, self.blend_strength * 1.5)
        elif self.consistency_mode == ConsistencyMode.ADAPTIVE:
            # Higher blend for lower consistency
            return self.blend_strength * (1.0 - report.consistency_score)
        return self.blend_strength

    def _copy_frame(self, src: Path, dst: Path) -> None:
        """Copy a frame file."""
        import shutil
        shutil.copy2(src, dst)

    def _blend_frames(self, frame1: Path, frame2: Path, output: Path, strength: float) -> None:
        """Blend two frames together."""
        # Placeholder - would use image blending
        # For now, just copy frame2
        self._copy_frame(frame2, output)

    def _extract_style_features(self, frame: Path) -> Dict[str, Any]:
        """Extract style features from a frame."""
        # Placeholder - would use style transfer network
        return {'source': str(frame)}

    def _apply_style(self, target: Path, style: Dict[str, Any], output: Path, strength: float) -> None:
        """Apply style features to a target frame."""
        # Placeholder - would use style transfer
        self._copy_frame(target, output)

    def clear_cache(self) -> None:
        """Clear all internal caches."""
        self._frame_cache.clear()
        self._flow_cache.clear()
        self._feature_cache.clear()

    def get_config(self) -> Dict[str, Any]:
        """Get current engine configuration."""
        return {
            'flow_method': self.flow_method.name,
            'consistency_mode': self.consistency_mode.name,
            'blend_strength': self.blend_strength,
            'motion_threshold': self.motion_threshold,
        }


__all__ = [
    'FlowMethod',
    'ConsistencyMode',
    'MotionVector',
    'FlowField',
    'ConsistencyReport',
    'TemporalConsistencyEngine',
]
