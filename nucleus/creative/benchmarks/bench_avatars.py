"""
Avatars Subsystem Benchmarks
============================

Benchmarks for 3D Gaussian Splatting (3DGS) and SMPL-X operations.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Callable, Any, Optional

from .bench_runner import BenchmarkSuite


@dataclass
class Vector3:
    """3D vector."""
    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> "Vector3":
        length = self.length()
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / length, self.y / length, self.z / length)

    @classmethod
    def random(cls, scale: float = 1.0) -> "Vector3":
        return cls(
            random.gauss(0, scale),
            random.gauss(0, scale),
            random.gauss(0, scale),
        )


@dataclass
class Quaternion:
    """Quaternion for rotations."""
    w: float
    x: float
    y: float
    z: float

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle: float) -> "Quaternion":
        axis = axis.normalize()
        half_angle = angle / 2
        s = math.sin(half_angle)
        return cls(
            w=math.cos(half_angle),
            x=axis.x * s,
            y=axis.y * s,
            z=axis.z * s,
        )

    @classmethod
    def random(cls) -> "Quaternion":
        """Random unit quaternion."""
        u1, u2, u3 = random.random(), random.random(), random.random()
        return cls(
            w=math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            x=math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            y=math.sqrt(u1) * math.sin(2 * math.pi * u3),
            z=math.sqrt(u1) * math.cos(2 * math.pi * u3),
        )

    def normalize(self) -> "Quaternion":
        length = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if length == 0:
            return Quaternion.identity()
        return Quaternion(
            self.w / length,
            self.x / length,
            self.y / length,
            self.z / length,
        )

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def rotate_vector(self, v: Vector3) -> Vector3:
        """Rotate vector by quaternion."""
        q_v = Quaternion(0, v.x, v.y, v.z)
        q_conj = Quaternion(self.w, -self.x, -self.y, -self.z)
        result = self * q_v * q_conj
        return Vector3(result.x, result.y, result.z)


@dataclass
class Gaussian3D:
    """3D Gaussian for splatting."""
    position: Vector3
    rotation: Quaternion
    scale: Vector3
    opacity: float
    color_sh: list[float]  # Spherical harmonics coefficients

    @classmethod
    def random(cls) -> "Gaussian3D":
        """Create random Gaussian."""
        return cls(
            position=Vector3.random(2.0),
            rotation=Quaternion.random(),
            scale=Vector3(
                abs(random.gauss(0.1, 0.05)),
                abs(random.gauss(0.1, 0.05)),
                abs(random.gauss(0.1, 0.05)),
            ),
            opacity=random.random(),
            color_sh=[random.gauss(0, 1) for _ in range(16 * 3)],  # 16 SH bands, RGB
        )

    def get_covariance_3d(self) -> list[list[float]]:
        """Compute 3D covariance matrix."""
        # Simplified covariance from scale and rotation
        S = [[self.scale.x, 0, 0], [0, self.scale.y, 0], [0, 0, self.scale.z]]

        # This is a simplified computation
        return [
            [self.scale.x ** 2, 0, 0],
            [0, self.scale.y ** 2, 0],
            [0, 0, self.scale.z ** 2],
        ]


@dataclass
class GaussianSplatCloud:
    """Collection of 3D Gaussians."""
    gaussians: list[Gaussian3D]

    @classmethod
    def random(cls, n_gaussians: int) -> "GaussianSplatCloud":
        """Create random Gaussian cloud."""
        return cls(gaussians=[Gaussian3D.random() for _ in range(n_gaussians)])

    def sort_by_depth(self, camera_pos: Vector3) -> "GaussianSplatCloud":
        """Sort Gaussians by distance to camera."""
        sorted_gaussians = sorted(
            self.gaussians,
            key=lambda g: (g.position - camera_pos).length(),
            reverse=True,  # Far to near for alpha blending
        )
        return GaussianSplatCloud(gaussians=sorted_gaussians)

    def cull_frustum(self, camera_pos: Vector3, fov: float, near: float, far: float) -> "GaussianSplatCloud":
        """Cull Gaussians outside view frustum."""
        culled = []
        for g in self.gaussians:
            dist = (g.position - camera_pos).length()
            if near <= dist <= far:
                culled.append(g)
        return GaussianSplatCloud(gaussians=culled)

    def merge(self, other: "GaussianSplatCloud") -> "GaussianSplatCloud":
        """Merge two Gaussian clouds."""
        return GaussianSplatCloud(gaussians=self.gaussians + other.gaussians)

    def densify(self, threshold: float = 0.01) -> "GaussianSplatCloud":
        """Densify Gaussians with large gradients."""
        new_gaussians = list(self.gaussians)

        for g in self.gaussians:
            # Simplified: split large Gaussians
            if g.scale.length() > threshold:
                # Create two smaller Gaussians
                offset = Vector3.random(0.1)
                g1 = Gaussian3D(
                    position=g.position + offset,
                    rotation=g.rotation,
                    scale=g.scale * 0.6,
                    opacity=g.opacity,
                    color_sh=g.color_sh.copy(),
                )
                g2 = Gaussian3D(
                    position=g.position - offset,
                    rotation=g.rotation,
                    scale=g.scale * 0.6,
                    opacity=g.opacity,
                    color_sh=g.color_sh.copy(),
                )
                new_gaussians.extend([g1, g2])

        return GaussianSplatCloud(gaussians=new_gaussians)


@dataclass
class SMPLXParams:
    """SMPL-X body model parameters."""
    body_pose: list[float]  # 63 values (21 joints * 3 axis-angle)
    hand_pose: list[float]  # 90 values (2 hands * 15 joints * 3)
    face_expression: list[float]  # 50 values
    betas: list[float]  # 10 shape parameters
    global_orient: list[float]  # 3 values
    translation: list[float]  # 3 values

    @classmethod
    def default(cls) -> "SMPLXParams":
        """Create default (rest) pose parameters."""
        return cls(
            body_pose=[0.0] * 63,
            hand_pose=[0.0] * 90,
            face_expression=[0.0] * 50,
            betas=[0.0] * 10,
            global_orient=[0.0, 0.0, 0.0],
            translation=[0.0, 0.0, 0.0],
        )

    @classmethod
    def random(cls) -> "SMPLXParams":
        """Create random pose parameters."""
        return cls(
            body_pose=[random.gauss(0, 0.3) for _ in range(63)],
            hand_pose=[random.gauss(0, 0.2) for _ in range(90)],
            face_expression=[random.gauss(0, 0.5) for _ in range(50)],
            betas=[random.gauss(0, 1.0) for _ in range(10)],
            global_orient=[random.gauss(0, 0.1) for _ in range(3)],
            translation=[random.gauss(0, 0.5) for _ in range(3)],
        )

    def interpolate(self, other: "SMPLXParams", t: float) -> "SMPLXParams":
        """Linear interpolation between poses."""
        def lerp_list(a: list[float], b: list[float]) -> list[float]:
            return [av * (1 - t) + bv * t for av, bv in zip(a, b)]

        return SMPLXParams(
            body_pose=lerp_list(self.body_pose, other.body_pose),
            hand_pose=lerp_list(self.hand_pose, other.hand_pose),
            face_expression=lerp_list(self.face_expression, other.face_expression),
            betas=lerp_list(self.betas, other.betas),
            global_orient=lerp_list(self.global_orient, other.global_orient),
            translation=lerp_list(self.translation, other.translation),
        )


class MockSMPLXExtractor:
    """Mock SMPL-X parameter extractor."""

    def __init__(self, use_hand_model: bool = True, use_face_model: bool = True):
        self.use_hand_model = use_hand_model
        self.use_face_model = use_face_model

    def extract_from_image(self, width: int, height: int) -> SMPLXParams:
        """Extract SMPL-X params from image (mock)."""
        # Simulate body detection and parameter estimation
        params = SMPLXParams.random()

        if not self.use_hand_model:
            params.hand_pose = [0.0] * 90
        if not self.use_face_model:
            params.face_expression = [0.0] * 50

        return params

    def forward_kinematics(self, params: SMPLXParams) -> list[Vector3]:
        """Compute joint positions from parameters."""
        # Mock: generate 55 joint positions
        joints = []
        for i in range(55):
            joint = Vector3(
                params.translation[0] + random.gauss(0, 0.3),
                params.translation[1] + i * 0.03,  # Roughly vertical chain
                params.translation[2] + random.gauss(0, 0.1),
            )
            joints.append(joint)
        return joints

    def generate_vertices(self, params: SMPLXParams) -> list[Vector3]:
        """Generate mesh vertices from parameters."""
        # Mock: generate 10475 vertices (actual SMPL-X vertex count)
        vertices = []
        base = Vector3(*params.translation)

        for i in range(10475):
            # Simplified: random offset based on betas
            shape_offset = sum(b * random.gauss(0, 0.01) for b in params.betas)
            vertex = Vector3(
                base.x + random.gauss(0, 0.3) + shape_offset,
                base.y + (i % 100) * 0.02,
                base.z + random.gauss(0, 0.2) + shape_offset,
            )
            vertices.append(vertex)

        return vertices


@dataclass
class DeformationResult:
    """Result of Gaussian deformation."""
    deformed_cloud: GaussianSplatCloud
    n_deformed: int
    max_displacement: float


class MockDeformableGaussians:
    """Mock deformable Gaussian splatting."""

    def __init__(self, base_cloud: GaussianSplatCloud):
        self.base_cloud = base_cloud

    def apply_smplx_deformation(
        self,
        params: SMPLXParams,
        blend_weight: float = 1.0,
    ) -> DeformationResult:
        """Apply SMPL-X based deformation to Gaussians."""
        deformed = []
        max_disp = 0.0

        for g in self.base_cloud.gaussians:
            # Simplified: offset based on SMPL-X translation and pose
            offset = Vector3(
                params.translation[0] * blend_weight,
                params.translation[1] * blend_weight,
                params.translation[2] * blend_weight,
            )

            new_pos = g.position + offset
            displacement = offset.length()
            max_disp = max(max_disp, displacement)

            deformed.append(Gaussian3D(
                position=new_pos,
                rotation=g.rotation,
                scale=g.scale,
                opacity=g.opacity,
                color_sh=g.color_sh,
            ))

        return DeformationResult(
            deformed_cloud=GaussianSplatCloud(deformed),
            n_deformed=len(deformed),
            max_displacement=max_disp,
        )

    def apply_blend_shapes(
        self,
        expression: list[float],
        blend_weight: float = 1.0,
    ) -> DeformationResult:
        """Apply facial blend shapes."""
        deformed = []
        max_disp = 0.0

        for g in self.base_cloud.gaussians:
            # Simplified: random offset based on expression weights
            expr_offset = sum(e * random.gauss(0, 0.01) for e in expression[:10])
            offset = Vector3(
                expr_offset * blend_weight,
                expr_offset * 0.5 * blend_weight,
                expr_offset * 0.2 * blend_weight,
            )

            new_pos = g.position + offset
            displacement = offset.length()
            max_disp = max(max_disp, displacement)

            deformed.append(Gaussian3D(
                position=new_pos,
                rotation=g.rotation,
                scale=g.scale,
                opacity=g.opacity,
                color_sh=g.color_sh,
            ))

        return DeformationResult(
            deformed_cloud=GaussianSplatCloud(deformed),
            n_deformed=len(deformed),
            max_displacement=max_disp,
        )


class AvatarsBenchmark(BenchmarkSuite):
    """Benchmark suite for avatars subsystem."""

    @property
    def name(self) -> str:
        return "avatars"

    @property
    def description(self) -> str:
        return "3DGS and SMPL-X benchmarks"

    def __init__(self):
        self._clouds: dict[str, GaussianSplatCloud] = {}
        self._params: list[SMPLXParams] = []
        self._extractor: Optional[MockSMPLXExtractor] = None
        self._deformable: Optional[MockDeformableGaussians] = None

    def setup(self) -> None:
        """Setup test data."""
        # Pre-generate Gaussian clouds
        self._clouds = {
            "tiny": GaussianSplatCloud.random(100),
            "small": GaussianSplatCloud.random(1000),
            "medium": GaussianSplatCloud.random(10000),
            "large": GaussianSplatCloud.random(50000),
        }

        # Pre-generate SMPL-X parameters
        self._params = [SMPLXParams.random() for _ in range(10)]

        self._extractor = MockSMPLXExtractor()
        self._deformable = MockDeformableGaussians(self._clouds["small"])

    def get_benchmarks(self) -> list[tuple[str, Callable[[], Any]]]:
        """Get all avatars benchmarks."""
        return [
            # 3DGS creation
            ("gaussian_create_single", self._gaussian_create_single),
            ("gaussian_cloud_tiny", self._gaussian_cloud_tiny),
            ("gaussian_cloud_small", self._gaussian_cloud_small),
            ("gaussian_cloud_medium", self._gaussian_cloud_medium),

            # 3DGS operations
            ("gaussian_sort_depth", self._gaussian_sort_depth),
            ("gaussian_cull_frustum", self._gaussian_cull_frustum),
            ("gaussian_merge_clouds", self._gaussian_merge_clouds),
            ("gaussian_densify", self._gaussian_densify),
            ("gaussian_covariance", self._gaussian_covariance),

            # SMPL-X parameters
            ("smplx_create_default", self._smplx_create_default),
            ("smplx_create_random", self._smplx_create_random),
            ("smplx_interpolate", self._smplx_interpolate),

            # SMPL-X extraction
            ("smplx_extract_from_image", self._smplx_extract_from_image),
            ("smplx_forward_kinematics", self._smplx_forward_kinematics),
            ("smplx_generate_vertices", self._smplx_generate_vertices),

            # Deformation
            ("deform_smplx_small", self._deform_smplx_small),
            ("deform_blend_shapes", self._deform_blend_shapes),
        ]

    def _gaussian_create_single(self) -> Gaussian3D:
        """Create single Gaussian."""
        return Gaussian3D.random()

    def _gaussian_cloud_tiny(self) -> GaussianSplatCloud:
        """Create tiny Gaussian cloud (100)."""
        return GaussianSplatCloud.random(100)

    def _gaussian_cloud_small(self) -> GaussianSplatCloud:
        """Create small Gaussian cloud (1000)."""
        return GaussianSplatCloud.random(1000)

    def _gaussian_cloud_medium(self) -> GaussianSplatCloud:
        """Create medium Gaussian cloud (10000)."""
        return GaussianSplatCloud.random(10000)

    def _gaussian_sort_depth(self) -> GaussianSplatCloud:
        """Sort Gaussians by depth."""
        camera = Vector3(0, 0, -5)
        return self._clouds["small"].sort_by_depth(camera)

    def _gaussian_cull_frustum(self) -> GaussianSplatCloud:
        """Cull Gaussians outside frustum."""
        camera = Vector3(0, 0, -5)
        return self._clouds["small"].cull_frustum(camera, fov=60, near=0.1, far=100)

    def _gaussian_merge_clouds(self) -> GaussianSplatCloud:
        """Merge two Gaussian clouds."""
        return self._clouds["tiny"].merge(self._clouds["tiny"])

    def _gaussian_densify(self) -> GaussianSplatCloud:
        """Densify Gaussian cloud."""
        return self._clouds["tiny"].densify(threshold=0.1)

    def _gaussian_covariance(self) -> list[list[list[float]]]:
        """Compute covariance matrices."""
        return [g.get_covariance_3d() for g in self._clouds["tiny"].gaussians]

    def _smplx_create_default(self) -> SMPLXParams:
        """Create default SMPL-X params."""
        return SMPLXParams.default()

    def _smplx_create_random(self) -> SMPLXParams:
        """Create random SMPL-X params."""
        return SMPLXParams.random()

    def _smplx_interpolate(self) -> SMPLXParams:
        """Interpolate between two poses."""
        return self._params[0].interpolate(self._params[1], 0.5)

    def _smplx_extract_from_image(self) -> SMPLXParams:
        """Extract SMPL-X from image."""
        return self._extractor.extract_from_image(512, 512)

    def _smplx_forward_kinematics(self) -> list[Vector3]:
        """Compute forward kinematics."""
        return self._extractor.forward_kinematics(self._params[0])

    def _smplx_generate_vertices(self) -> list[Vector3]:
        """Generate SMPL-X mesh vertices."""
        return self._extractor.generate_vertices(self._params[0])

    def _deform_smplx_small(self) -> DeformationResult:
        """Apply SMPL-X deformation."""
        return self._deformable.apply_smplx_deformation(self._params[0])

    def _deform_blend_shapes(self) -> DeformationResult:
        """Apply blend shape deformation."""
        return self._deformable.apply_blend_shapes(self._params[0].face_expression)
