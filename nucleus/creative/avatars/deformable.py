"""
Deformable Gaussian Avatars
===========================

Deformable 3D Gaussian Splatting for animatable human avatars.
Combines SMPL-X body model with 3DGS representation to enable
pose-dependent deformation of the Gaussian cloud.

Based on techniques from:
- "Animatable Gaussians" (Li et al., 2023)
- "Human Gaussian Splatting" (Moreau et al., 2023)
- "GaussianAvatars" (Qian et al., 2024)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
from enum import Enum, auto
from pathlib import Path
import numpy as np
import numpy.typing as npt
from datetime import datetime, timezone
import uuid
import json

from .smplx_extractor import SMPLXParams, BodyPart
from .gaussian_splatting import GaussianSplatCloud, Gaussian3D, GaussianCloudOperations


class DeformationType(Enum):
    """Types of deformation."""
    LINEAR_BLEND_SKINNING = auto()  # LBS from SMPL-X
    DUAL_QUATERNION = auto()        # DQS for better volume preservation
    DELTA_BLEND = auto()            # Neural delta blending
    NEURAL_BLEND_FIELD = auto()     # Full neural deformation field


class BindingType(Enum):
    """How Gaussians are bound to the body."""
    NEAREST_VERTEX = auto()     # Bind to nearest SMPL-X vertex
    NEAREST_JOINT = auto()      # Bind to nearest skeleton joint
    SKINNING_WEIGHTS = auto()   # Use SMPL-X skinning weights
    LEARNED = auto()            # Learned binding weights


@dataclass
class DeformationResult:
    """
    Result of a deformation operation.

    Contains the deformed Gaussian cloud along with diagnostic
    information about the deformation quality.

    Attributes:
        cloud: Deformed Gaussian cloud.
        source_params: Source pose parameters.
        target_params: Target pose parameters.
        deformation_type: Type of deformation applied.
        quality_score: Quality metric (0-1).
        processing_time_ms: Time taken for deformation.
        diagnostics: Additional diagnostic information.
    """
    cloud: GaussianSplatCloud
    source_params: SMPLXParams
    target_params: SMPLXParams
    deformation_type: DeformationType = DeformationType.LINEAR_BLEND_SKINNING
    quality_score: float = 1.0
    processing_time_ms: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if deformation result is valid."""
        return (
            self.cloud is not None and
            len(self.cloud) > 0 and
            self.quality_score > 0.5
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "num_gaussians": len(self.cloud),
            "source_pose_id": self.source_params.id,
            "target_pose_id": self.target_params.id,
            "deformation_type": self.deformation_type.name,
            "quality_score": self.quality_score,
            "processing_time_ms": self.processing_time_ms,
            "diagnostics": self.diagnostics,
        }


@dataclass
class SkinningWeights:
    """
    Skinning weights for Gaussian-to-skeleton binding.

    Attributes:
        weights: NxJ array of skinning weights.
        joint_indices: NxK array of joint indices for sparse weights.
        k_nearest: Number of joints per Gaussian (for sparse).
    """
    weights: npt.NDArray[np.float32]
    joint_indices: Optional[npt.NDArray[np.int32]] = None
    k_nearest: int = 4

    def __post_init__(self) -> None:
        """Normalize weights."""
        self.weights = np.asarray(self.weights, dtype=np.float32)
        # Ensure weights sum to 1
        row_sums = self.weights.sum(axis=1, keepdims=True)
        self.weights = self.weights / (row_sums + 1e-8)

    def get_weights_for_gaussian(self, idx: int) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """
        Get joint indices and weights for a specific Gaussian.

        Args:
            idx: Gaussian index.

        Returns:
            Tuple of (joint_indices, weights).
        """
        if self.joint_indices is not None:
            return self.joint_indices[idx], self.weights[idx, :self.k_nearest]
        else:
            # Dense weights - return all non-zero
            w = self.weights[idx]
            nonzero = np.where(w > 1e-6)[0]
            return nonzero.astype(np.int32), w[nonzero]


@dataclass
class DeformableGaussiansConfig:
    """Configuration for deformable Gaussians."""
    deformation_type: DeformationType = DeformationType.LINEAR_BLEND_SKINNING
    binding_type: BindingType = BindingType.SKINNING_WEIGHTS
    k_nearest_joints: int = 4
    use_pose_dependent_color: bool = True
    use_pose_dependent_scale: bool = True
    use_neural_deltas: bool = False
    delta_mlp_layers: int = 3
    delta_mlp_hidden: int = 64
    smoothing_iterations: int = 0


class DeformableGaussians:
    """
    Deformable 3D Gaussian avatar representation.

    Combines a canonical-space Gaussian cloud with skinning information
    to enable pose-driven deformation. The avatar can be animated by
    providing target SMPL-X pose parameters.

    Attributes:
        canonical_cloud: Gaussian cloud in canonical (T/A) pose.
        canonical_params: SMPL-X parameters for canonical pose.
        skinning_weights: Per-Gaussian skinning weights.
        config: Deformation configuration.

    Example:
        >>> # Create avatar from point cloud
        >>> avatar = DeformableGaussians.from_point_cloud(
        ...     points, colors,
        ...     smplx_params=canonical_pose,
        ...     smplx_vertices=vertices
        ... )
        >>> # Deform to new pose
        >>> result = avatar.deform(target_pose)
        >>> deformed_cloud = result.cloud
    """

    def __init__(
        self,
        canonical_cloud: GaussianSplatCloud,
        canonical_params: SMPLXParams,
        skinning_weights: Optional[SkinningWeights] = None,
        config: Optional[DeformableGaussiansConfig] = None,
    ):
        """
        Initialize deformable Gaussian avatar.

        Args:
            canonical_cloud: Gaussians in canonical pose.
            canonical_params: SMPL-X parameters for canonical pose.
            skinning_weights: Optional pre-computed skinning weights.
            config: Deformation configuration.
        """
        self.canonical_cloud = canonical_cloud
        self.canonical_params = canonical_params
        self.skinning_weights = skinning_weights
        self.config = config or DeformableGaussiansConfig()

        # Cached joint transforms for current pose
        self._cached_transforms: Optional[npt.NDArray[np.float32]] = None
        self._cached_pose_id: Optional[str] = None

        # Canonical joint positions (computed from SMPL-X)
        self._canonical_joints: Optional[npt.NDArray[np.float32]] = None

        # Neural components (if using learned deformation)
        self._delta_network = None
        self._pose_encoder = None

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians in the avatar."""
        return len(self.canonical_cloud)

    @property
    def num_joints(self) -> int:
        """Number of skeleton joints."""
        return self.canonical_params.num_body_joints + 2  # Body + hands

    def initialize_skinning(
        self,
        smplx_vertices: Optional[npt.NDArray[np.float32]] = None,
        smplx_weights: Optional[npt.NDArray[np.float32]] = None,
    ) -> None:
        """
        Initialize skinning weights from SMPL-X model.

        Args:
            smplx_vertices: SMPL-X mesh vertices in canonical pose.
            smplx_weights: SMPL-X skinning weights (V x J).
        """
        if self.skinning_weights is not None:
            return

        n = len(self.canonical_cloud)
        num_joints = self.num_joints

        if smplx_vertices is not None and smplx_weights is not None:
            # Transfer weights from nearest SMPL-X vertices
            weights = self._transfer_weights_from_mesh(
                self.canonical_cloud.positions,
                smplx_vertices,
                smplx_weights,
            )
        else:
            # Compute weights based on distance to joints
            weights = self._compute_distance_based_weights()

        self.skinning_weights = SkinningWeights(
            weights=weights,
            k_nearest=self.config.k_nearest_joints,
        )

    def _transfer_weights_from_mesh(
        self,
        gaussian_positions: npt.NDArray[np.float32],
        vertices: npt.NDArray[np.float32],
        vertex_weights: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """
        Transfer skinning weights from mesh vertices to Gaussians.

        Args:
            gaussian_positions: Gaussian positions (N x 3).
            vertices: Mesh vertices (V x 3).
            vertex_weights: Vertex skinning weights (V x J).

        Returns:
            Gaussian skinning weights (N x J).
        """
        n = len(gaussian_positions)
        num_joints = vertex_weights.shape[1]

        # Find nearest vertices for each Gaussian
        weights = np.zeros((n, num_joints), dtype=np.float32)

        for i, pos in enumerate(gaussian_positions):
            # Compute distances to all vertices
            dists = np.linalg.norm(vertices - pos, axis=1)

            # Get k nearest vertices
            k = min(4, len(vertices))
            nearest_idx = np.argpartition(dists, k)[:k]
            nearest_dists = dists[nearest_idx]

            # Inverse distance weighting
            inv_dists = 1.0 / (nearest_dists + 1e-6)
            interp_weights = inv_dists / inv_dists.sum()

            # Interpolate skinning weights
            for j, (idx, w) in enumerate(zip(nearest_idx, interp_weights)):
                weights[i] += w * vertex_weights[idx]

        return weights

    def _compute_distance_based_weights(self) -> npt.NDArray[np.float32]:
        """
        Compute skinning weights based on distance to joints.

        Returns:
            Skinning weights (N x J).
        """
        n = len(self.canonical_cloud)
        num_joints = self.num_joints

        # Get approximate joint positions from canonical pose
        # This is a placeholder - real implementation would use SMPL-X forward kinematics
        joint_positions = self._get_canonical_joint_positions()
        self._canonical_joints = joint_positions

        weights = np.zeros((n, num_joints), dtype=np.float32)

        for i, pos in enumerate(self.canonical_cloud.positions):
            # Compute distances to all joints
            dists = np.linalg.norm(joint_positions - pos, axis=1)

            # Get k nearest joints
            k = min(self.config.k_nearest_joints, num_joints)
            nearest_idx = np.argpartition(dists, k)[:k]
            nearest_dists = dists[nearest_idx]

            # Gaussian falloff weighting
            sigma = np.mean(nearest_dists) + 0.1
            w = np.exp(-nearest_dists ** 2 / (2 * sigma ** 2))
            w = w / (w.sum() + 1e-8)

            weights[i, nearest_idx] = w

        return weights

    def _get_canonical_joint_positions(self) -> npt.NDArray[np.float32]:
        """
        Get joint positions in canonical pose.

        Returns:
            Joint positions (J x 3).
        """
        if self._canonical_joints is not None:
            return self._canonical_joints

        num_joints = self.num_joints

        # Approximate T-pose joint positions
        # Real implementation would use SMPL-X forward kinematics
        joints = np.zeros((num_joints, 3), dtype=np.float32)

        # Pelvis at origin
        joints[0] = [0, 0, 0]

        # Spine
        joints[3] = [0, 0.1, 0]   # Spine1
        joints[6] = [0, 0.2, 0]   # Spine2
        joints[9] = [0, 0.3, 0]   # Spine3
        joints[12] = [0, 0.4, 0]  # Neck
        joints[15] = [0, 0.5, 0]  # Head

        # Left leg
        joints[1] = [-0.1, -0.05, 0]  # Left hip
        joints[4] = [-0.1, -0.5, 0]   # Left knee
        joints[7] = [-0.1, -0.9, 0]   # Left ankle
        joints[10] = [-0.1, -0.95, 0.05]  # Left foot

        # Right leg
        joints[2] = [0.1, -0.05, 0]   # Right hip
        joints[5] = [0.1, -0.5, 0]    # Right knee
        joints[8] = [0.1, -0.9, 0]    # Right ankle
        joints[11] = [0.1, -0.95, 0.05]  # Right foot

        # Left arm
        joints[13] = [-0.15, 0.35, 0]  # Left collar
        joints[16] = [-0.25, 0.35, 0]  # Left shoulder
        joints[18] = [-0.5, 0.35, 0]   # Left elbow
        joints[20] = [-0.75, 0.35, 0]  # Left wrist

        # Right arm
        joints[14] = [0.15, 0.35, 0]   # Right collar
        joints[17] = [0.25, 0.35, 0]   # Right shoulder
        joints[19] = [0.5, 0.35, 0]    # Right elbow
        joints[21] = [0.75, 0.35, 0]   # Right wrist

        self._canonical_joints = joints
        return joints

    def deform(
        self,
        target_params: SMPLXParams,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> DeformationResult:
        """
        Deform the avatar to a target pose.

        Args:
            target_params: Target SMPL-X parameters.
            progress_callback: Optional progress callback.

        Returns:
            DeformationResult with deformed cloud.
        """
        import time
        start_time = time.perf_counter()

        if progress_callback:
            progress_callback("initializing", 0.0)

        # Ensure skinning is initialized
        self.initialize_skinning()

        if progress_callback:
            progress_callback("computing_transforms", 20.0)

        # Compute joint transforms from canonical to target pose
        transforms = self._compute_joint_transforms(target_params)

        if progress_callback:
            progress_callback("deforming_positions", 40.0)

        # Apply deformation based on type
        if self.config.deformation_type == DeformationType.LINEAR_BLEND_SKINNING:
            deformed_positions, deformed_rotations = self._apply_lbs(transforms)
        elif self.config.deformation_type == DeformationType.DUAL_QUATERNION:
            deformed_positions, deformed_rotations = self._apply_dqs(transforms)
        else:
            deformed_positions, deformed_rotations = self._apply_lbs(transforms)

        if progress_callback:
            progress_callback("applying_deltas", 60.0)

        # Apply pose-dependent deltas if enabled
        if self.config.use_neural_deltas and self._delta_network is not None:
            position_deltas = self._compute_neural_deltas(target_params)
            deformed_positions = deformed_positions + position_deltas

        if progress_callback:
            progress_callback("building_cloud", 80.0)

        # Build deformed cloud
        deformed_cloud = GaussianSplatCloud(
            positions=deformed_positions,
            scales=self._compute_deformed_scales(target_params),
            rotations=deformed_rotations,
            opacities=self.canonical_cloud.opacities.copy(),
            sh_coeffs=self._compute_deformed_sh(target_params),
            metadata={
                "deformed_from": self.canonical_params.id,
                "target_pose": target_params.id,
            },
        )

        # Compute quality score
        quality_score = self._compute_quality_score(deformed_cloud)

        if progress_callback:
            progress_callback("complete", 100.0)

        processing_time = (time.perf_counter() - start_time) * 1000

        return DeformationResult(
            cloud=deformed_cloud,
            source_params=self.canonical_params,
            target_params=target_params,
            deformation_type=self.config.deformation_type,
            quality_score=quality_score,
            processing_time_ms=processing_time,
            diagnostics={
                "num_gaussians": len(deformed_cloud),
                "mean_displacement": float(np.mean(np.linalg.norm(
                    deformed_positions - self.canonical_cloud.positions, axis=1
                ))),
            },
        )

    def _compute_joint_transforms(
        self,
        target_params: SMPLXParams,
    ) -> npt.NDArray[np.float32]:
        """
        Compute per-joint transformation matrices.

        Args:
            target_params: Target pose parameters.

        Returns:
            Joint transforms (J x 4 x 4).
        """
        num_joints = self.num_joints
        transforms = np.zeros((num_joints, 4, 4), dtype=np.float32)

        # Get canonical joint positions
        canonical_joints = self._get_canonical_joint_positions()

        # Compute transforms for each joint
        for j in range(min(num_joints, len(target_params.body_pose))):
            # Get rotation from pose
            axis_angle = target_params.body_pose[j] - self.canonical_params.body_pose[j]
            R = self._axis_angle_to_matrix(axis_angle)

            # Build 4x4 transform
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R

            # Translation from joint position
            if j < len(canonical_joints):
                joint_pos = canonical_joints[j]
                # Rotate around joint position
                T[:3, 3] = joint_pos - R @ joint_pos

            transforms[j] = T

        # Apply global transform
        global_R = self._axis_angle_to_matrix(
            target_params.global_orient - self.canonical_params.global_orient
        )
        global_t = target_params.transl - self.canonical_params.transl

        global_T = np.eye(4, dtype=np.float32)
        global_T[:3, :3] = global_R
        global_T[:3, 3] = global_t

        # Compose global transform with each joint
        for j in range(num_joints):
            transforms[j] = global_T @ transforms[j]

        return transforms

    def _apply_lbs(
        self,
        transforms: npt.NDArray[np.float32],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Apply Linear Blend Skinning.

        Args:
            transforms: Joint transforms (J x 4 x 4).

        Returns:
            Tuple of (deformed_positions, deformed_rotations).
        """
        n = len(self.canonical_cloud)
        num_joints = transforms.shape[0]

        positions = self.canonical_cloud.positions
        rotations = self.canonical_cloud.rotations

        deformed_positions = np.zeros_like(positions)
        deformed_rotations = np.zeros_like(rotations)

        # Get skinning weights
        weights = self.skinning_weights.weights

        for i in range(n):
            # Blend transforms
            blended_T = np.zeros((4, 4), dtype=np.float32)
            for j in range(min(num_joints, weights.shape[1])):
                w = weights[i, j]
                if w > 1e-6:
                    blended_T += w * transforms[j]

            # Apply to position (homogeneous coordinates)
            pos_h = np.array([*positions[i], 1], dtype=np.float32)
            deformed_pos_h = blended_T @ pos_h
            deformed_positions[i] = deformed_pos_h[:3]

            # Apply to rotation
            blended_R = blended_T[:3, :3]
            # Normalize rotation matrix
            U, _, Vt = np.linalg.svd(blended_R)
            blended_R = U @ Vt

            # Convert canonical quaternion to matrix, apply blend, convert back
            R_canonical = self._quaternion_to_matrix(rotations[i])
            R_deformed = blended_R @ R_canonical
            deformed_rotations[i] = self._matrix_to_quaternion(R_deformed)

        return deformed_positions, deformed_rotations

    def _apply_dqs(
        self,
        transforms: npt.NDArray[np.float32],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Apply Dual Quaternion Skinning for better volume preservation.

        Args:
            transforms: Joint transforms (J x 4 x 4).

        Returns:
            Tuple of (deformed_positions, deformed_rotations).
        """
        n = len(self.canonical_cloud)
        num_joints = transforms.shape[0]

        positions = self.canonical_cloud.positions
        rotations = self.canonical_cloud.rotations

        # Convert transforms to dual quaternions
        dqs = []
        for j in range(num_joints):
            R = transforms[j, :3, :3]
            t = transforms[j, :3, 3]
            dq = self._matrix_to_dual_quaternion(R, t)
            dqs.append(dq)
        dqs = np.array(dqs, dtype=np.float32)

        deformed_positions = np.zeros_like(positions)
        deformed_rotations = np.zeros_like(rotations)

        weights = self.skinning_weights.weights

        for i in range(n):
            # Blend dual quaternions
            blended_dq = np.zeros(8, dtype=np.float32)
            for j in range(min(num_joints, weights.shape[1])):
                w = weights[i, j]
                if w > 1e-6:
                    # Handle quaternion antipodality
                    if np.dot(dqs[j, :4], blended_dq[:4]) < 0:
                        blended_dq -= w * dqs[j]
                    else:
                        blended_dq += w * dqs[j]

            # Normalize dual quaternion
            norm = np.linalg.norm(blended_dq[:4])
            if norm > 1e-8:
                blended_dq = blended_dq / norm

            # Apply dual quaternion to position
            R, t = self._dual_quaternion_to_matrix(blended_dq)
            deformed_positions[i] = R @ positions[i] + t

            # Apply to rotation
            R_canonical = self._quaternion_to_matrix(rotations[i])
            R_deformed = R @ R_canonical
            deformed_rotations[i] = self._matrix_to_quaternion(R_deformed)

        return deformed_positions, deformed_rotations

    def _compute_deformed_scales(
        self,
        target_params: SMPLXParams,
    ) -> npt.NDArray[np.float32]:
        """
        Compute pose-dependent scales.

        Args:
            target_params: Target pose parameters.

        Returns:
            Deformed scales (N x 3).
        """
        if not self.config.use_pose_dependent_scale:
            return self.canonical_cloud.scales.copy()

        # For now, return canonical scales
        # Real implementation would learn pose-dependent scale changes
        return self.canonical_cloud.scales.copy()

    def _compute_deformed_sh(
        self,
        target_params: SMPLXParams,
    ) -> npt.NDArray[np.float32]:
        """
        Compute pose-dependent SH coefficients.

        Args:
            target_params: Target pose parameters.

        Returns:
            Deformed SH coefficients.
        """
        if not self.config.use_pose_dependent_color:
            return self.canonical_cloud.sh_coeffs.copy()

        # For now, return canonical SH
        # Real implementation would learn pose-dependent color changes
        return self.canonical_cloud.sh_coeffs.copy()

    def _compute_neural_deltas(
        self,
        target_params: SMPLXParams,
    ) -> npt.NDArray[np.float32]:
        """
        Compute neural position deltas.

        Args:
            target_params: Target pose parameters.

        Returns:
            Position deltas (N x 3).
        """
        # Placeholder - real implementation would use an MLP
        return np.zeros_like(self.canonical_cloud.positions)

    def _compute_quality_score(
        self,
        deformed_cloud: GaussianSplatCloud,
    ) -> float:
        """
        Compute quality score for deformation.

        Args:
            deformed_cloud: Deformed Gaussian cloud.

        Returns:
            Quality score (0-1).
        """
        # Check for NaN/Inf
        if np.any(~np.isfinite(deformed_cloud.positions)):
            return 0.0

        # Check for excessive deformation
        max_displacement = np.max(np.linalg.norm(
            deformed_cloud.positions - self.canonical_cloud.positions, axis=1
        ))
        if max_displacement > 10.0:  # Unreasonable displacement
            return 0.5

        return 1.0

    def _axis_angle_to_matrix(self, axis_angle: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Convert axis-angle to rotation matrix."""
        angle = np.linalg.norm(axis_angle)
        if angle < 1e-8:
            return np.eye(3, dtype=np.float32)

        axis = axis_angle / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ], dtype=np.float32)

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R.astype(np.float32)

    def _quaternion_to_matrix(self, q: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Convert quaternion (w,x,y,z) to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float32)

    def _matrix_to_quaternion(self, R: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Convert rotation matrix to quaternion (w,x,y,z)."""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z], dtype=np.float32)

    def _matrix_to_dual_quaternion(
        self,
        R: npt.NDArray[np.float32],
        t: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Convert rotation matrix and translation to dual quaternion."""
        # Real part (rotation)
        q_r = self._matrix_to_quaternion(R)

        # Dual part (translation)
        t_quat = np.array([0, t[0], t[1], t[2]], dtype=np.float32)
        q_d = 0.5 * self._quaternion_multiply(t_quat, q_r)

        return np.concatenate([q_r, q_d])

    def _dual_quaternion_to_matrix(
        self,
        dq: npt.NDArray[np.float32],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Convert dual quaternion to rotation matrix and translation."""
        q_r = dq[:4]
        q_d = dq[4:]

        # Rotation
        R = self._quaternion_to_matrix(q_r)

        # Translation
        t_quat = 2.0 * self._quaternion_multiply(q_d, self._quaternion_conjugate(q_r))
        t = t_quat[1:4]

        return R, t

    def _quaternion_multiply(
        self,
        q1: npt.NDArray[np.float32],
        q2: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float32)

    def _quaternion_conjugate(
        self,
        q: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Compute quaternion conjugate."""
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    @classmethod
    def from_point_cloud(
        cls,
        points: npt.NDArray[np.float32],
        colors: Optional[npt.NDArray[np.float32]] = None,
        smplx_params: Optional[SMPLXParams] = None,
        smplx_vertices: Optional[npt.NDArray[np.float32]] = None,
        smplx_weights: Optional[npt.NDArray[np.float32]] = None,
        config: Optional[DeformableGaussiansConfig] = None,
    ) -> "DeformableGaussians":
        """
        Create deformable avatar from point cloud.

        Args:
            points: Point cloud positions (N x 3).
            colors: Optional RGB colors (N x 3).
            smplx_params: SMPL-X parameters for canonical pose.
            smplx_vertices: SMPL-X mesh vertices for weight transfer.
            smplx_weights: SMPL-X skinning weights.
            config: Deformation configuration.

        Returns:
            DeformableGaussians instance.
        """
        # Initialize canonical Gaussian cloud
        cloud = GaussianCloudOperations.from_point_cloud(
            points=points,
            colors=colors,
            initial_scale=0.01,
            sh_degree=0,
        )

        # Use provided or create default canonical params
        if smplx_params is None:
            from .smplx_extractor import create_t_pose
            smplx_params = create_t_pose()

        # Create deformable avatar
        avatar = cls(
            canonical_cloud=cloud,
            canonical_params=smplx_params,
            config=config,
        )

        # Initialize skinning
        avatar.initialize_skinning(smplx_vertices, smplx_weights)

        return avatar

    def save(self, path: Union[str, Path]) -> None:
        """
        Save avatar to directory.

        Args:
            path: Output directory path.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save canonical cloud
        self.canonical_cloud.save(path / "canonical_cloud.ply")

        # Save canonical params
        self.canonical_params.save(path / "canonical_params.json")

        # Save skinning weights
        if self.skinning_weights is not None:
            np.savez(
                path / "skinning_weights.npz",
                weights=self.skinning_weights.weights,
                joint_indices=self.skinning_weights.joint_indices,
            )

        # Save config
        config_dict = {
            "deformation_type": self.config.deformation_type.name,
            "binding_type": self.config.binding_type.name,
            "k_nearest_joints": self.config.k_nearest_joints,
            "use_pose_dependent_color": self.config.use_pose_dependent_color,
            "use_pose_dependent_scale": self.config.use_pose_dependent_scale,
            "use_neural_deltas": self.config.use_neural_deltas,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeformableGaussians":
        """
        Load avatar from directory.

        Args:
            path: Input directory path.

        Returns:
            DeformableGaussians instance.
        """
        path = Path(path)

        # Load canonical cloud
        cloud = GaussianSplatCloud.load(path / "canonical_cloud.ply")

        # Load canonical params
        params = SMPLXParams.load(path / "canonical_params.json")

        # Load skinning weights
        skinning = None
        weights_path = path / "skinning_weights.npz"
        if weights_path.exists():
            data = np.load(weights_path)
            skinning = SkinningWeights(
                weights=data["weights"],
                joint_indices=data.get("joint_indices"),
            )

        # Load config
        config = DeformableGaussiansConfig()
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config.deformation_type = DeformationType[config_dict.get("deformation_type", "LINEAR_BLEND_SKINNING")]
            config.binding_type = BindingType[config_dict.get("binding_type", "SKINNING_WEIGHTS")]
            config.k_nearest_joints = config_dict.get("k_nearest_joints", 4)
            config.use_pose_dependent_color = config_dict.get("use_pose_dependent_color", True)
            config.use_pose_dependent_scale = config_dict.get("use_pose_dependent_scale", True)
            config.use_neural_deltas = config_dict.get("use_neural_deltas", False)

        return cls(
            canonical_cloud=cloud,
            canonical_params=params,
            skinning_weights=skinning,
            config=config,
        )
