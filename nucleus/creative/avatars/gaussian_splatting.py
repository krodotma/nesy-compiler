"""
3D Gaussian Splatting
=====================

Implementation of 3D Gaussian Splatting for novel view synthesis and
avatar representation. Based on "3D Gaussian Splatting for Real-Time
Radiance Field Rendering" (Kerbl et al., 2023).

Each 3D Gaussian is defined by:
- Position (mean): 3D center point
- Covariance: 3x3 covariance matrix (stored as 6 parameters)
- Opacity: Scalar alpha value
- Spherical Harmonics: Color representation for view-dependent effects

References:
- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple, Iterator
from enum import Enum, auto
from pathlib import Path
import numpy as np
import numpy.typing as npt
from datetime import datetime, timezone
import uuid
import json
import struct


class SHDegree(Enum):
    """Spherical harmonics degree."""
    DEGREE_0 = 0  # 1 coefficient per channel (constant)
    DEGREE_1 = 1  # 4 coefficients per channel
    DEGREE_2 = 2  # 9 coefficients per channel
    DEGREE_3 = 3  # 16 coefficients per channel

    @property
    def num_coefficients(self) -> int:
        """Number of SH coefficients per color channel."""
        return (self.value + 1) ** 2


@dataclass
class Gaussian3D:
    """
    A single 3D Gaussian primitive.

    The Gaussian is parameterized by:
    - position: 3D mean (x, y, z)
    - scale: 3D scale factors (sx, sy, sz) in log space
    - rotation: Quaternion (w, x, y, z)
    - opacity: Scalar opacity in logit space
    - sh_coeffs: Spherical harmonics for view-dependent color

    Attributes:
        position: 3D position of the Gaussian center.
        scale: Log-space scale factors.
        rotation: Quaternion rotation (w, x, y, z).
        opacity_logit: Logit-space opacity value.
        sh_coeffs: Spherical harmonics coefficients (degree^2 x 3).
        id: Unique identifier.
    """
    position: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    scale: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    rotation: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([1, 0, 0, 0], dtype=np.float32)  # Identity quaternion
    )
    opacity_logit: float = 0.0
    sh_coeffs: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((1, 3), dtype=np.float32)  # Degree 0
    )
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self) -> None:
        """Validate and convert arrays."""
        self.position = np.asarray(self.position, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        self.rotation = np.asarray(self.rotation, dtype=np.float32)
        self.sh_coeffs = np.asarray(self.sh_coeffs, dtype=np.float32)

        # Normalize quaternion
        self.rotation = self.rotation / (np.linalg.norm(self.rotation) + 1e-8)

    @property
    def opacity(self) -> float:
        """Get opacity in [0, 1] range from logit."""
        return 1.0 / (1.0 + np.exp(-self.opacity_logit))

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set opacity, converting to logit space."""
        value = np.clip(value, 1e-6, 1 - 1e-6)
        self.opacity_logit = np.log(value / (1 - value))

    @property
    def scale_actual(self) -> npt.NDArray[np.float32]:
        """Get actual scale from log-space scale."""
        return np.exp(self.scale)

    @scale_actual.setter
    def scale_actual(self, value: npt.NDArray[np.float32]) -> None:
        """Set scale from actual values, converting to log space."""
        self.scale = np.log(np.maximum(value, 1e-8)).astype(np.float32)

    @property
    def sh_degree(self) -> int:
        """Infer SH degree from number of coefficients."""
        num_coeffs = self.sh_coeffs.shape[0]
        return int(np.sqrt(num_coeffs)) - 1

    @property
    def covariance(self) -> npt.NDArray[np.float32]:
        """
        Compute 3x3 covariance matrix from scale and rotation.

        Covariance = R * S * S^T * R^T
        where R is the rotation matrix and S is the diagonal scale matrix.
        """
        R = self._quaternion_to_rotation_matrix()
        S = np.diag(self.scale_actual)
        return R @ S @ S.T @ R.T

    def _quaternion_to_rotation_matrix(self) -> npt.NDArray[np.float32]:
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = self.rotation

        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float32)

    def get_color(self, view_direction: Optional[npt.NDArray[np.float32]] = None) -> npt.NDArray[np.float32]:
        """
        Get RGB color, optionally considering view direction.

        Args:
            view_direction: Normalized view direction (3,).

        Returns:
            RGB color (3,) in [0, 1] range.
        """
        if view_direction is None or self.sh_degree == 0:
            # Use DC (constant) term only
            dc = self.sh_coeffs[0]
            # SH to RGB conversion (with offset)
            return np.clip(dc * 0.282095 + 0.5, 0, 1)

        # Evaluate full SH (simplified for degree 0-3)
        color = self._evaluate_sh(view_direction)
        return np.clip(color + 0.5, 0, 1)

    def _evaluate_sh(self, direction: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Evaluate spherical harmonics for a given direction."""
        x, y, z = direction

        # SH basis functions
        basis = [
            0.282095,  # Y_0^0
        ]

        if self.sh_degree >= 1:
            basis.extend([
                -0.488603 * y,    # Y_1^{-1}
                0.488603 * z,     # Y_1^0
                -0.488603 * x,    # Y_1^1
            ])

        if self.sh_degree >= 2:
            basis.extend([
                1.092548 * x * y,              # Y_2^{-2}
                -1.092548 * y * z,             # Y_2^{-1}
                0.315392 * (3*z*z - 1),        # Y_2^0
                -1.092548 * x * z,             # Y_2^1
                0.546274 * (x*x - y*y),        # Y_2^2
            ])

        if self.sh_degree >= 3:
            basis.extend([
                -0.590044 * y * (3*x*x - y*y),         # Y_3^{-3}
                2.890611 * x * y * z,                   # Y_3^{-2}
                -0.457046 * y * (5*z*z - 1),           # Y_3^{-1}
                0.373176 * z * (5*z*z - 3),            # Y_3^0
                -0.457046 * x * (5*z*z - 1),           # Y_3^1
                1.445306 * z * (x*x - y*y),            # Y_3^2
                -0.590044 * x * (x*x - 3*y*y),         # Y_3^3
            ])

        basis = np.array(basis[:len(self.sh_coeffs)], dtype=np.float32)

        # Compute color from SH
        color = np.sum(self.sh_coeffs * basis[:, np.newaxis], axis=0)
        return color.astype(np.float32)

    def clone(self) -> "Gaussian3D":
        """Create a deep copy of this Gaussian."""
        return Gaussian3D(
            position=self.position.copy(),
            scale=self.scale.copy(),
            rotation=self.rotation.copy(),
            opacity_logit=self.opacity_logit,
            sh_coeffs=self.sh_coeffs.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "scale": self.scale.tolist(),
            "rotation": self.rotation.tolist(),
            "opacity_logit": self.opacity_logit,
            "sh_coeffs": self.sh_coeffs.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Gaussian3D":
        """Create from dictionary representation."""
        return cls(
            position=np.array(data["position"]),
            scale=np.array(data["scale"]),
            rotation=np.array(data["rotation"]),
            opacity_logit=data["opacity_logit"],
            sh_coeffs=np.array(data["sh_coeffs"]),
            id=data.get("id", str(uuid.uuid4())[:8]),
        )


@dataclass
class GaussianSplatCloud:
    """
    A collection of 3D Gaussians representing a scene or object.

    Optimized for batched operations with contiguous arrays for
    positions, scales, rotations, opacities, and SH coefficients.

    Attributes:
        positions: Nx3 array of Gaussian centers.
        scales: Nx3 array of log-space scales.
        rotations: Nx4 array of quaternions.
        opacities: N array of logit-space opacities.
        sh_coeffs: NxCx3 array of SH coefficients.
        metadata: Additional metadata.
    """
    positions: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    scales: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    rotations: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.float32)
    )
    opacities: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    sh_coeffs: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 1, 3), dtype=np.float32)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate arrays."""
        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        self.rotations = np.asarray(self.rotations, dtype=np.float32)
        self.opacities = np.asarray(self.opacities, dtype=np.float32)
        self.sh_coeffs = np.asarray(self.sh_coeffs, dtype=np.float32)

    def __len__(self) -> int:
        """Number of Gaussians in the cloud."""
        return len(self.positions)

    def __getitem__(self, idx: Union[int, slice, npt.NDArray]) -> Union[Gaussian3D, "GaussianSplatCloud"]:
        """Get Gaussian(s) by index."""
        if isinstance(idx, int):
            return Gaussian3D(
                position=self.positions[idx],
                scale=self.scales[idx],
                rotation=self.rotations[idx],
                opacity_logit=float(self.opacities[idx]),
                sh_coeffs=self.sh_coeffs[idx],
            )
        else:
            return GaussianSplatCloud(
                positions=self.positions[idx],
                scales=self.scales[idx],
                rotations=self.rotations[idx],
                opacities=self.opacities[idx],
                sh_coeffs=self.sh_coeffs[idx],
                metadata=self.metadata.copy(),
            )

    def __iter__(self) -> Iterator[Gaussian3D]:
        """Iterate over Gaussians."""
        for i in range(len(self)):
            yield self[i]

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians."""
        return len(self)

    @property
    def sh_degree(self) -> int:
        """Spherical harmonics degree."""
        if len(self.sh_coeffs) == 0:
            return 0
        return int(np.sqrt(self.sh_coeffs.shape[1])) - 1

    @property
    def bounds(self) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Get axis-aligned bounding box.

        Returns:
            Tuple of (min_point, max_point).
        """
        if len(self) == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        return self.positions.min(axis=0), self.positions.max(axis=0)

    @property
    def center(self) -> npt.NDArray[np.float32]:
        """Get center of the bounding box."""
        min_pt, max_pt = self.bounds
        return (min_pt + max_pt) / 2

    def add(self, gaussian: Gaussian3D) -> None:
        """
        Add a Gaussian to the cloud.

        Args:
            gaussian: Gaussian to add.
        """
        self.positions = np.vstack([self.positions, gaussian.position[np.newaxis]])
        self.scales = np.vstack([self.scales, gaussian.scale[np.newaxis]])
        self.rotations = np.vstack([self.rotations, gaussian.rotation[np.newaxis]])
        self.opacities = np.append(self.opacities, gaussian.opacity_logit)

        # Handle SH coefficient size mismatch
        if len(self.sh_coeffs) == 0 or self.sh_coeffs.shape[1] == gaussian.sh_coeffs.shape[0]:
            self.sh_coeffs = np.vstack([self.sh_coeffs, gaussian.sh_coeffs[np.newaxis]])
        else:
            # Pad or truncate to match existing degree
            target_coeffs = self.sh_coeffs.shape[1]
            new_coeffs = np.zeros((1, target_coeffs, 3), dtype=np.float32)
            copy_len = min(target_coeffs, len(gaussian.sh_coeffs))
            new_coeffs[0, :copy_len] = gaussian.sh_coeffs[:copy_len]
            self.sh_coeffs = np.vstack([self.sh_coeffs, new_coeffs])

    def remove(self, indices: Union[int, List[int], npt.NDArray]) -> None:
        """
        Remove Gaussians by index.

        Args:
            indices: Index or indices to remove.
        """
        if isinstance(indices, int):
            indices = [indices]
        mask = np.ones(len(self), dtype=bool)
        mask[indices] = False

        self.positions = self.positions[mask]
        self.scales = self.scales[mask]
        self.rotations = self.rotations[mask]
        self.opacities = self.opacities[mask]
        self.sh_coeffs = self.sh_coeffs[mask]

    def merge(self, other: "GaussianSplatCloud") -> "GaussianSplatCloud":
        """
        Merge with another cloud.

        Args:
            other: Cloud to merge with.

        Returns:
            New merged cloud.
        """
        if len(self) == 0:
            return other.clone()
        if len(other) == 0:
            return self.clone()

        # Handle SH degree mismatch
        max_degree = max(self.sh_degree, other.sh_degree)
        num_coeffs = (max_degree + 1) ** 2

        def pad_sh(sh: npt.NDArray, target: int) -> npt.NDArray:
            if sh.shape[1] == target:
                return sh
            result = np.zeros((len(sh), target, 3), dtype=np.float32)
            result[:, :sh.shape[1]] = sh
            return result

        return GaussianSplatCloud(
            positions=np.vstack([self.positions, other.positions]),
            scales=np.vstack([self.scales, other.scales]),
            rotations=np.vstack([self.rotations, other.rotations]),
            opacities=np.concatenate([self.opacities, other.opacities]),
            sh_coeffs=np.vstack([
                pad_sh(self.sh_coeffs, num_coeffs),
                pad_sh(other.sh_coeffs, num_coeffs),
            ]),
            metadata={**self.metadata, **other.metadata, "merged": True},
        )

    def clone(self) -> "GaussianSplatCloud":
        """Create a deep copy."""
        return GaussianSplatCloud(
            positions=self.positions.copy(),
            scales=self.scales.copy(),
            rotations=self.rotations.copy(),
            opacities=self.opacities.copy(),
            sh_coeffs=self.sh_coeffs.copy(),
            metadata=self.metadata.copy(),
        )

    def filter_by_opacity(self, min_opacity: float = 0.01) -> "GaussianSplatCloud":
        """
        Filter out low-opacity Gaussians.

        Args:
            min_opacity: Minimum opacity threshold.

        Returns:
            Filtered cloud.
        """
        # Convert logit to opacity
        actual_opacities = 1.0 / (1.0 + np.exp(-self.opacities))
        mask = actual_opacities >= min_opacity
        return self[mask]

    def filter_by_bounds(
        self,
        min_point: npt.NDArray[np.float32],
        max_point: npt.NDArray[np.float32],
    ) -> "GaussianSplatCloud":
        """
        Filter Gaussians within bounding box.

        Args:
            min_point: Minimum corner (3,).
            max_point: Maximum corner (3,).

        Returns:
            Filtered cloud.
        """
        mask = np.all(
            (self.positions >= min_point) & (self.positions <= max_point),
            axis=1
        )
        return self[mask]

    def transform(
        self,
        rotation: Optional[npt.NDArray[np.float32]] = None,
        translation: Optional[npt.NDArray[np.float32]] = None,
        scale: float = 1.0,
    ) -> "GaussianSplatCloud":
        """
        Apply rigid transformation to all Gaussians.

        Args:
            rotation: 3x3 rotation matrix.
            translation: 3D translation vector.
            scale: Uniform scale factor.

        Returns:
            Transformed cloud.
        """
        result = self.clone()

        # Apply scale
        if scale != 1.0:
            result.positions *= scale
            result.scales += np.log(scale)

        # Apply rotation
        if rotation is not None:
            result.positions = (rotation @ result.positions.T).T

            # Also rotate each Gaussian's orientation
            for i in range(len(result)):
                quat = result.rotations[i]
                R_gauss = _quaternion_to_matrix(quat)
                R_new = rotation @ R_gauss
                result.rotations[i] = _matrix_to_quaternion(R_new)

        # Apply translation
        if translation is not None:
            result.positions += translation

        return result

    def save(self, path: Union[str, Path]) -> None:
        """
        Save cloud to file.

        Args:
            path: Output path (.ply or .json).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".ply":
            self._save_ply(path)
        else:
            self._save_json(path)

    def _save_ply(self, path: Path) -> None:
        """Save as PLY file (3DGS format)."""
        n = len(self)
        num_sh = self.sh_coeffs.shape[1] if len(self.sh_coeffs) > 0 else 1

        # Build PLY header
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "property float opacity",
        ]

        # Add SH properties
        for i in range(num_sh):
            for c in ["r", "g", "b"]:
                header.append(f"property float sh_{c}_{i}")

        header.append("end_header")

        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n").encode())

            for i in range(n):
                # Position
                f.write(struct.pack("fff", *self.positions[i]))
                # Scale
                f.write(struct.pack("fff", *self.scales[i]))
                # Rotation
                f.write(struct.pack("ffff", *self.rotations[i]))
                # Opacity
                f.write(struct.pack("f", self.opacities[i]))
                # SH coefficients
                for j in range(num_sh):
                    f.write(struct.pack("fff", *self.sh_coeffs[i, j]))

    def _save_json(self, path: Path) -> None:
        """Save as JSON file."""
        data = {
            "num_gaussians": len(self),
            "sh_degree": self.sh_degree,
            "positions": self.positions.tolist(),
            "scales": self.scales.tolist(),
            "rotations": self.rotations.tolist(),
            "opacities": self.opacities.tolist(),
            "sh_coeffs": self.sh_coeffs.tolist(),
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GaussianSplatCloud":
        """
        Load cloud from file.

        Args:
            path: Input path (.ply or .json).

        Returns:
            Loaded cloud.
        """
        path = Path(path)

        if path.suffix == ".ply":
            return cls._load_ply(path)
        else:
            return cls._load_json(path)

    @classmethod
    def _load_ply(cls, path: Path) -> "GaussianSplatCloud":
        """Load from PLY file."""
        with open(path, "rb") as f:
            # Parse header
            header_lines = []
            while True:
                line = f.readline().decode().strip()
                header_lines.append(line)
                if line == "end_header":
                    break

            # Parse vertex count
            n = 0
            for line in header_lines:
                if line.startswith("element vertex"):
                    n = int(line.split()[-1])
                    break

            # Count SH properties
            sh_count = sum(1 for line in header_lines if "property float sh_" in line) // 3

            # Read binary data
            positions = np.zeros((n, 3), dtype=np.float32)
            scales = np.zeros((n, 3), dtype=np.float32)
            rotations = np.zeros((n, 4), dtype=np.float32)
            opacities = np.zeros(n, dtype=np.float32)
            sh_coeffs = np.zeros((n, max(sh_count, 1), 3), dtype=np.float32)

            for i in range(n):
                positions[i] = struct.unpack("fff", f.read(12))
                scales[i] = struct.unpack("fff", f.read(12))
                rotations[i] = struct.unpack("ffff", f.read(16))
                opacities[i] = struct.unpack("f", f.read(4))[0]
                for j in range(sh_count):
                    sh_coeffs[i, j] = struct.unpack("fff", f.read(12))

        return cls(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
        )

    @classmethod
    def _load_json(cls, path: Path) -> "GaussianSplatCloud":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        return cls(
            positions=np.array(data["positions"]),
            scales=np.array(data["scales"]),
            rotations=np.array(data["rotations"]),
            opacities=np.array(data["opacities"]),
            sh_coeffs=np.array(data["sh_coeffs"]),
            metadata=data.get("metadata", {}),
        )


class GaussianCloudOperations:
    """
    Operations on Gaussian splat clouds.

    Provides methods for:
    - Initialization from point clouds
    - Densification and pruning
    - Level-of-detail generation
    - Compression
    """

    @staticmethod
    def from_point_cloud(
        points: npt.NDArray[np.float32],
        colors: Optional[npt.NDArray[np.float32]] = None,
        normals: Optional[npt.NDArray[np.float32]] = None,
        initial_scale: float = 0.01,
        sh_degree: int = 0,
    ) -> GaussianSplatCloud:
        """
        Initialize Gaussians from a point cloud.

        Args:
            points: Nx3 point positions.
            colors: Nx3 RGB colors (optional).
            normals: Nx3 surface normals (optional).
            initial_scale: Initial Gaussian scale.
            sh_degree: Spherical harmonics degree.

        Returns:
            Initialized GaussianSplatCloud.
        """
        n = len(points)

        # Initialize positions
        positions = np.asarray(points, dtype=np.float32)

        # Initialize scales (log-space)
        scales = np.full((n, 3), np.log(initial_scale), dtype=np.float32)

        # Initialize rotations (identity quaternion)
        rotations = np.zeros((n, 4), dtype=np.float32)
        rotations[:, 0] = 1.0  # w=1, x=y=z=0

        # If normals provided, align rotations
        if normals is not None:
            for i, normal in enumerate(normals):
                rotations[i] = _align_quaternion_to_normal(normal)

        # Initialize opacities (0 in logit space = 0.5 opacity)
        opacities = np.zeros(n, dtype=np.float32)

        # Initialize SH coefficients
        num_coeffs = (sh_degree + 1) ** 2
        sh_coeffs = np.zeros((n, num_coeffs, 3), dtype=np.float32)

        # Set DC term from colors if provided
        if colors is not None:
            # Convert RGB to SH DC coefficient
            # SH_DC = (color - 0.5) / 0.282095
            colors = np.asarray(colors, dtype=np.float32)
            if colors.max() > 1:
                colors = colors / 255.0
            sh_coeffs[:, 0] = (colors - 0.5) / 0.282095

        return GaussianSplatCloud(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
            metadata={"source": "point_cloud", "initial_scale": initial_scale},
        )

    @staticmethod
    def densify(
        cloud: GaussianSplatCloud,
        gradient_threshold: float = 0.0002,
        scale_threshold: float = 0.01,
        split_ratio: float = 0.5,
    ) -> GaussianSplatCloud:
        """
        Densify cloud by splitting large Gaussians.

        This implements the adaptive density control from the 3DGS paper,
        which splits Gaussians with high positional gradients during training.

        Args:
            cloud: Input cloud.
            gradient_threshold: Threshold for splitting.
            scale_threshold: Minimum scale to consider for splitting.
            split_ratio: Ratio of offset for split Gaussians.

        Returns:
            Densified cloud.
        """
        result = cloud.clone()

        # Find Gaussians to split (large scale)
        actual_scales = np.exp(result.scales)
        max_scales = np.max(actual_scales, axis=1)
        split_mask = max_scales > scale_threshold

        if not np.any(split_mask):
            return result

        indices = np.where(split_mask)[0]

        # Split each selected Gaussian into two
        new_positions = []
        new_scales = []
        new_rotations = []
        new_opacities = []
        new_sh_coeffs = []

        for i in indices:
            pos = result.positions[i]
            scale = result.scales[i]
            rot = result.rotations[i]
            opacity = result.opacities[i]
            sh = result.sh_coeffs[i]

            # Compute split direction (along largest axis)
            actual_scale = np.exp(scale)
            split_axis = np.argmax(actual_scale)
            offset = np.zeros(3)
            offset[split_axis] = actual_scale[split_axis] * split_ratio

            # Create two new Gaussians
            for sign in [-1, 1]:
                new_positions.append(pos + sign * offset)
                new_scales.append(scale - np.log(2))  # Half the scale
                new_rotations.append(rot)
                new_opacities.append(opacity)
                new_sh_coeffs.append(sh)

        # Remove original Gaussians
        result.remove(indices)

        # Add split Gaussians
        if new_positions:
            result.positions = np.vstack([result.positions, np.array(new_positions)])
            result.scales = np.vstack([result.scales, np.array(new_scales)])
            result.rotations = np.vstack([result.rotations, np.array(new_rotations)])
            result.opacities = np.concatenate([result.opacities, np.array(new_opacities)])
            result.sh_coeffs = np.vstack([result.sh_coeffs, np.array(new_sh_coeffs)])

        result.metadata["densified"] = True
        return result

    @staticmethod
    def prune(
        cloud: GaussianSplatCloud,
        min_opacity: float = 0.01,
        max_scale: float = 1.0,
    ) -> GaussianSplatCloud:
        """
        Prune Gaussians that are too transparent or too large.

        Args:
            cloud: Input cloud.
            min_opacity: Minimum opacity threshold.
            max_scale: Maximum scale threshold.

        Returns:
            Pruned cloud.
        """
        # Compute actual opacities and scales
        actual_opacities = 1.0 / (1.0 + np.exp(-cloud.opacities))
        actual_scales = np.exp(cloud.scales)
        max_scales = np.max(actual_scales, axis=1)

        # Keep Gaussians meeting criteria
        mask = (actual_opacities >= min_opacity) & (max_scales <= max_scale)

        result = cloud[mask]
        result.metadata["pruned"] = True
        result.metadata["removed_count"] = np.sum(~mask)
        return result

    @staticmethod
    def generate_lod(
        cloud: GaussianSplatCloud,
        levels: int = 4,
        reduction_factor: float = 0.5,
    ) -> List[GaussianSplatCloud]:
        """
        Generate level-of-detail versions of the cloud.

        Args:
            cloud: Input cloud.
            levels: Number of LOD levels.
            reduction_factor: Gaussian reduction per level.

        Returns:
            List of clouds from highest to lowest detail.
        """
        lods = [cloud.clone()]

        current = cloud
        for level in range(1, levels):
            # Reduce number of Gaussians
            n_target = int(len(current) * reduction_factor)
            if n_target < 10:
                break

            # Simple reduction: keep highest opacity Gaussians
            actual_opacities = 1.0 / (1.0 + np.exp(-current.opacities))
            indices = np.argsort(actual_opacities)[-n_target:]

            reduced = current[indices]
            reduced.metadata["lod_level"] = level

            # Increase scale to compensate for fewer Gaussians
            reduced.scales += np.log(1 / reduction_factor) / 3

            lods.append(reduced)
            current = reduced

        return lods

    @staticmethod
    def compress(
        cloud: GaussianSplatCloud,
        position_bits: int = 16,
        scale_bits: int = 8,
        rotation_bits: int = 8,
        sh_bits: int = 8,
    ) -> bytes:
        """
        Compress cloud to binary format.

        Args:
            cloud: Input cloud.
            position_bits: Bits per position component.
            scale_bits: Bits per scale component.
            rotation_bits: Bits per rotation component.
            sh_bits: Bits per SH coefficient.

        Returns:
            Compressed bytes.
        """
        from io import BytesIO

        buffer = BytesIO()

        # Write header
        n = len(cloud)
        sh_degree = cloud.sh_degree

        buffer.write(struct.pack("<I", n))  # num gaussians
        buffer.write(struct.pack("<B", sh_degree))  # SH degree
        buffer.write(struct.pack("<B", position_bits))
        buffer.write(struct.pack("<B", scale_bits))
        buffer.write(struct.pack("<B", rotation_bits))
        buffer.write(struct.pack("<B", sh_bits))

        # Compute bounding box for position quantization
        min_pos = cloud.positions.min(axis=0)
        max_pos = cloud.positions.max(axis=0)
        buffer.write(struct.pack("<fff", *min_pos))
        buffer.write(struct.pack("<fff", *max_pos))

        # Quantize and write positions
        pos_range = max_pos - min_pos + 1e-6
        pos_normalized = (cloud.positions - min_pos) / pos_range
        pos_quantized = (pos_normalized * ((1 << position_bits) - 1)).astype(np.uint16)
        buffer.write(pos_quantized.tobytes())

        # Write scales (8-bit quantized)
        scale_min, scale_max = cloud.scales.min(), cloud.scales.max()
        buffer.write(struct.pack("<ff", scale_min, scale_max))
        scale_normalized = (cloud.scales - scale_min) / (scale_max - scale_min + 1e-6)
        scale_quantized = (scale_normalized * 255).astype(np.uint8)
        buffer.write(scale_quantized.tobytes())

        # Write rotations (8-bit per component)
        rot_quantized = ((cloud.rotations + 1) * 127.5).astype(np.uint8)
        buffer.write(rot_quantized.tobytes())

        # Write opacities (8-bit)
        opacity_quantized = ((cloud.opacities + 10) * 12.75).astype(np.uint8)  # Assuming range [-10, 10]
        buffer.write(opacity_quantized.tobytes())

        # Write SH coefficients (8-bit)
        sh_min, sh_max = cloud.sh_coeffs.min(), cloud.sh_coeffs.max()
        buffer.write(struct.pack("<ff", sh_min, sh_max))
        sh_normalized = (cloud.sh_coeffs - sh_min) / (sh_max - sh_min + 1e-6)
        sh_quantized = (sh_normalized * 255).astype(np.uint8)
        buffer.write(sh_quantized.tobytes())

        return buffer.getvalue()

    @staticmethod
    def decompress(data: bytes) -> GaussianSplatCloud:
        """
        Decompress cloud from binary format.

        Args:
            data: Compressed bytes.

        Returns:
            Decompressed cloud.
        """
        from io import BytesIO

        buffer = BytesIO(data)

        # Read header
        n = struct.unpack("<I", buffer.read(4))[0]
        sh_degree = struct.unpack("<B", buffer.read(1))[0]
        position_bits = struct.unpack("<B", buffer.read(1))[0]
        scale_bits = struct.unpack("<B", buffer.read(1))[0]
        rotation_bits = struct.unpack("<B", buffer.read(1))[0]
        sh_bits = struct.unpack("<B", buffer.read(1))[0]

        # Read bounding box
        min_pos = np.array(struct.unpack("<fff", buffer.read(12)))
        max_pos = np.array(struct.unpack("<fff", buffer.read(12)))

        # Read positions
        pos_quantized = np.frombuffer(buffer.read(n * 3 * 2), dtype=np.uint16).reshape(n, 3)
        pos_range = max_pos - min_pos
        positions = (pos_quantized / ((1 << position_bits) - 1)) * pos_range + min_pos

        # Read scales
        scale_min, scale_max = struct.unpack("<ff", buffer.read(8))
        scale_quantized = np.frombuffer(buffer.read(n * 3), dtype=np.uint8).reshape(n, 3)
        scales = (scale_quantized / 255.0) * (scale_max - scale_min) + scale_min

        # Read rotations
        rot_quantized = np.frombuffer(buffer.read(n * 4), dtype=np.uint8).reshape(n, 4)
        rotations = (rot_quantized / 127.5) - 1

        # Read opacities
        opacity_quantized = np.frombuffer(buffer.read(n), dtype=np.uint8)
        opacities = (opacity_quantized / 12.75) - 10

        # Read SH coefficients
        sh_min, sh_max = struct.unpack("<ff", buffer.read(8))
        num_coeffs = (sh_degree + 1) ** 2
        sh_size = n * num_coeffs * 3
        sh_quantized = np.frombuffer(buffer.read(sh_size), dtype=np.uint8).reshape(n, num_coeffs, 3)
        sh_coeffs = (sh_quantized / 255.0) * (sh_max - sh_min) + sh_min

        return GaussianSplatCloud(
            positions=positions.astype(np.float32),
            scales=scales.astype(np.float32),
            rotations=rotations.astype(np.float32),
            opacities=opacities.astype(np.float32),
            sh_coeffs=sh_coeffs.astype(np.float32),
            metadata={"decompressed": True},
        )

    @staticmethod
    def render_depth(
        cloud: GaussianSplatCloud,
        camera_position: npt.NDArray[np.float32],
        camera_direction: npt.NDArray[np.float32],
        image_width: int,
        image_height: int,
    ) -> npt.NDArray[np.float32]:
        """
        Render a simple depth map from the Gaussian cloud.

        This is a simplified placeholder for the full differentiable
        rendering pipeline used in 3DGS training.

        Args:
            cloud: Gaussian cloud to render.
            camera_position: Camera position (3,).
            camera_direction: Camera look direction (3,).
            image_width: Output image width.
            image_height: Output image height.

        Returns:
            Depth image (HxW float32).
        """
        # Compute depths (distance along view direction)
        relative_pos = cloud.positions - camera_position
        depths = np.dot(relative_pos, camera_direction)

        # Create simple depth image (placeholder)
        depth_image = np.full((image_height, image_width), np.inf, dtype=np.float32)

        # Project Gaussians to image (simplified orthographic)
        right = np.cross(camera_direction, np.array([0, 1, 0]))
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, camera_direction)

        for i in range(len(cloud)):
            if depths[i] < 0:
                continue

            pos = relative_pos[i]
            x = int((np.dot(pos, right) + 1) * image_width / 2)
            y = int((np.dot(pos, up) + 1) * image_height / 2)

            if 0 <= x < image_width and 0 <= y < image_height:
                depth_image[y, x] = min(depth_image[y, x], depths[i])

        return depth_image


# Helper functions

def _quaternion_to_matrix(q: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def _matrix_to_quaternion(R: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z)."""
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


def _align_quaternion_to_normal(normal: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Create quaternion that aligns Z-axis with the given normal."""
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    z_axis = np.array([0, 0, 1], dtype=np.float32)

    # Handle parallel/anti-parallel cases
    dot = np.dot(z_axis, normal)
    if dot > 0.9999:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    if dot < -0.9999:
        return np.array([0, 1, 0, 0], dtype=np.float32)

    # Compute rotation axis and angle
    axis = np.cross(z_axis, normal)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np.arccos(np.clip(dot, -1, 1))

    # Axis-angle to quaternion
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)

    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float32)
