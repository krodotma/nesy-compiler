"""
SMPL-X Body Model Extraction
============================

Extracts SMPL-X body model parameters from images and video frames.
SMPL-X (SMPL eXpressive) is a unified body model with shape, pose,
hand pose, and facial expression parameters.

References:
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- PyTorch3D: https://pytorch3d.org/
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


class BodyPart(Enum):
    """Body part enumeration for SMPL-X model."""
    PELVIS = 0
    LEFT_HIP = 1
    RIGHT_HIP = 2
    SPINE1 = 3
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    SPINE2 = 6
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    SPINE3 = 9
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    NECK = 12
    LEFT_COLLAR = 13
    RIGHT_COLLAR = 14
    HEAD = 15
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21


class HandJoint(Enum):
    """Hand joint enumeration for SMPL-X MANO hands."""
    WRIST = 0
    INDEX1 = 1
    INDEX2 = 2
    INDEX3 = 3
    MIDDLE1 = 4
    MIDDLE2 = 5
    MIDDLE3 = 6
    PINKY1 = 7
    PINKY2 = 8
    PINKY3 = 9
    RING1 = 10
    RING2 = 11
    RING3 = 12
    THUMB1 = 13
    THUMB2 = 14
    THUMB3 = 15


@dataclass
class SMPLXParams:
    """
    SMPL-X body model parameters.

    SMPL-X combines body, face, and hands in a unified model:
    - Body pose: 21 joints x 3 (axis-angle) = 63 parameters
    - Hand pose: 15 joints x 3 x 2 (left/right) = 90 parameters
    - Face expression: 10 coefficients (FLAME model)
    - Shape (betas): 10-300 coefficients for body shape
    - Global orientation: 3 parameters (root rotation)
    - Translation: 3 parameters (root position)

    All rotations are in axis-angle representation.

    Attributes:
        id: Unique identifier for this parameter set.
        body_pose: Body joint rotations (21 joints x 3).
        left_hand_pose: Left hand joint rotations (15 joints x 3).
        right_hand_pose: Right hand joint rotations (15 joints x 3).
        jaw_pose: Jaw rotation (1 x 3).
        leye_pose: Left eye rotation (1 x 3).
        reye_pose: Right eye rotation (1 x 3).
        expression: Face expression coefficients (10).
        betas: Body shape coefficients (10-300).
        global_orient: Root orientation (1 x 3).
        transl: Root translation (3).
        scale: Global scale factor.
        confidence: Per-joint confidence scores.
        timestamp: Extraction timestamp.
        metadata: Additional metadata.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Body pose (21 joints x 3 axis-angle)
    body_pose: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((21, 3), dtype=np.float32)
    )

    # Hand poses (15 joints x 3 each)
    left_hand_pose: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((15, 3), dtype=np.float32)
    )
    right_hand_pose: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((15, 3), dtype=np.float32)
    )

    # Face poses
    jaw_pose: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    leye_pose: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    reye_pose: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    # Expression coefficients (FLAME model)
    expression: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(10, dtype=np.float32)
    )

    # Shape coefficients
    betas: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(10, dtype=np.float32)
    )

    # Global parameters
    global_orient: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    transl: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    scale: float = 1.0

    # Confidence and metadata
    confidence: Optional[npt.NDArray[np.float32]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and convert arrays to proper types."""
        # Ensure all arrays are float32
        self.body_pose = np.asarray(self.body_pose, dtype=np.float32)
        self.left_hand_pose = np.asarray(self.left_hand_pose, dtype=np.float32)
        self.right_hand_pose = np.asarray(self.right_hand_pose, dtype=np.float32)
        self.jaw_pose = np.asarray(self.jaw_pose, dtype=np.float32)
        self.leye_pose = np.asarray(self.leye_pose, dtype=np.float32)
        self.reye_pose = np.asarray(self.reye_pose, dtype=np.float32)
        self.expression = np.asarray(self.expression, dtype=np.float32)
        self.betas = np.asarray(self.betas, dtype=np.float32)
        self.global_orient = np.asarray(self.global_orient, dtype=np.float32)
        self.transl = np.asarray(self.transl, dtype=np.float32)

    @property
    def num_body_joints(self) -> int:
        """Number of body joints."""
        return self.body_pose.shape[0]

    @property
    def num_hand_joints(self) -> int:
        """Number of hand joints per hand."""
        return self.left_hand_pose.shape[0]

    @property
    def num_betas(self) -> int:
        """Number of shape coefficients."""
        return len(self.betas)

    @property
    def num_expression_coeffs(self) -> int:
        """Number of expression coefficients."""
        return len(self.expression)

    def get_full_pose(self) -> npt.NDArray[np.float32]:
        """
        Get the full pose vector (body + hands + face).

        Returns:
            Concatenated pose array (63 + 45 + 45 + 9 = 162 params).
        """
        return np.concatenate([
            self.body_pose.flatten(),
            self.left_hand_pose.flatten(),
            self.right_hand_pose.flatten(),
            self.jaw_pose,
            self.leye_pose,
            self.reye_pose,
        ])

    def get_joint_rotation(self, joint: Union[BodyPart, int]) -> npt.NDArray[np.float32]:
        """
        Get rotation for a specific body joint.

        Args:
            joint: BodyPart enum or joint index.

        Returns:
            Axis-angle rotation (3,).
        """
        idx = joint.value if isinstance(joint, BodyPart) else joint
        if idx >= self.num_body_joints:
            raise IndexError(f"Joint index {idx} out of range")
        return self.body_pose[idx]

    def set_joint_rotation(
        self,
        joint: Union[BodyPart, int],
        rotation: npt.NDArray[np.float32]
    ) -> None:
        """
        Set rotation for a specific body joint.

        Args:
            joint: BodyPart enum or joint index.
            rotation: Axis-angle rotation (3,).
        """
        idx = joint.value if isinstance(joint, BodyPart) else joint
        if idx >= self.num_body_joints:
            raise IndexError(f"Joint index {idx} out of range")
        self.body_pose[idx] = np.asarray(rotation, dtype=np.float32)

    def interpolate(self, other: "SMPLXParams", t: float) -> "SMPLXParams":
        """
        Linearly interpolate between this and another parameter set.

        Args:
            other: Target SMPLXParams.
            t: Interpolation factor (0=self, 1=other).

        Returns:
            Interpolated SMPLXParams.
        """
        t = np.clip(t, 0.0, 1.0)

        return SMPLXParams(
            body_pose=self.body_pose * (1 - t) + other.body_pose * t,
            left_hand_pose=self.left_hand_pose * (1 - t) + other.left_hand_pose * t,
            right_hand_pose=self.right_hand_pose * (1 - t) + other.right_hand_pose * t,
            jaw_pose=self.jaw_pose * (1 - t) + other.jaw_pose * t,
            leye_pose=self.leye_pose * (1 - t) + other.leye_pose * t,
            reye_pose=self.reye_pose * (1 - t) + other.reye_pose * t,
            expression=self.expression * (1 - t) + other.expression * t,
            betas=self.betas * (1 - t) + other.betas * t,
            global_orient=self.global_orient * (1 - t) + other.global_orient * t,
            transl=self.transl * (1 - t) + other.transl * t,
            scale=self.scale * (1 - t) + other.scale * t,
            metadata={"interpolated": True, "t": t},
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with all parameters as lists.
        """
        return {
            "id": self.id,
            "body_pose": self.body_pose.tolist(),
            "left_hand_pose": self.left_hand_pose.tolist(),
            "right_hand_pose": self.right_hand_pose.tolist(),
            "jaw_pose": self.jaw_pose.tolist(),
            "leye_pose": self.leye_pose.tolist(),
            "reye_pose": self.reye_pose.tolist(),
            "expression": self.expression.tolist(),
            "betas": self.betas.tolist(),
            "global_orient": self.global_orient.tolist(),
            "transl": self.transl.tolist(),
            "scale": self.scale,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SMPLXParams":
        """
        Create from dictionary representation.

        Args:
            data: Dictionary with parameter data.

        Returns:
            SMPLXParams instance.
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            body_pose=np.array(data.get("body_pose", np.zeros((21, 3)))),
            left_hand_pose=np.array(data.get("left_hand_pose", np.zeros((15, 3)))),
            right_hand_pose=np.array(data.get("right_hand_pose", np.zeros((15, 3)))),
            jaw_pose=np.array(data.get("jaw_pose", np.zeros(3))),
            leye_pose=np.array(data.get("leye_pose", np.zeros(3))),
            reye_pose=np.array(data.get("reye_pose", np.zeros(3))),
            expression=np.array(data.get("expression", np.zeros(10))),
            betas=np.array(data.get("betas", np.zeros(10))),
            global_orient=np.array(data.get("global_orient", np.zeros(3))),
            transl=np.array(data.get("transl", np.zeros(3))),
            scale=data.get("scale", 1.0),
            timestamp=timestamp or datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save parameters to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SMPLXParams":
        """
        Load parameters from JSON file.

        Args:
            path: Input file path.

        Returns:
            SMPLXParams instance.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ExtractionConfig:
    """Configuration for SMPL-X extraction."""
    # Detection settings
    detect_hands: bool = True
    detect_face: bool = True
    use_gender: Optional[str] = None  # 'male', 'female', or None for neutral

    # Shape settings
    num_betas: int = 10
    num_expression_coeffs: int = 10

    # Optimization settings
    num_iterations: int = 100
    learning_rate: float = 0.01
    regularization: float = 0.001

    # Quality settings
    min_confidence: float = 0.5
    smooth_poses: bool = True
    temporal_window: int = 5

    # Output settings
    output_format: str = "smplx"  # 'smplx' or 'smpl'
    save_mesh: bool = False


@dataclass
class ExtractionResult:
    """Result of SMPL-X extraction."""
    params: SMPLXParams
    vertices: Optional[npt.NDArray[np.float32]] = None
    faces: Optional[npt.NDArray[np.int32]] = None
    joints_2d: Optional[npt.NDArray[np.float32]] = None
    joints_3d: Optional[npt.NDArray[np.float32]] = None
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SMPLXExtractor:
    """
    Extracts SMPL-X body model parameters from images and video.

    This class provides methods to detect human bodies in images and
    fit the SMPL-X model to obtain body pose, shape, and expression
    parameters.

    Attributes:
        config: Extraction configuration.
        model_path: Path to SMPL-X model files.
        device: Computation device ('cpu' or 'cuda').

    Example:
        >>> extractor = SMPLXExtractor()
        >>> image = load_image("person.jpg")  # HxWx3 uint8
        >>> result = extractor.extract(image)
        >>> print(result.params.body_pose.shape)  # (21, 3)
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        model_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        """
        Initialize the SMPL-X extractor.

        Args:
            config: Extraction configuration.
            model_path: Path to SMPL-X model files.
            device: Computation device ('cpu' or 'cuda').
        """
        self.config = config or ExtractionConfig()
        self.model_path = model_path or Path("/models/smplx")
        self.device = device

        # Lazy-loaded model components
        self._body_model = None
        self._pose_detector = None
        self._hand_detector = None
        self._face_detector = None

    def _ensure_model_loaded(self) -> None:
        """Ensure the SMPL-X model is loaded."""
        if self._body_model is not None:
            return

        # In a real implementation, this would load the actual SMPL-X model.
        # For now, we create a placeholder that allows the API to work.
        self._body_model = {
            "loaded": True,
            "gender": self.config.use_gender or "neutral",
            "num_betas": self.config.num_betas,
            "num_expression_coeffs": self.config.num_expression_coeffs,
        }

    def extract(
        self,
        image: npt.NDArray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        prior_params: Optional[SMPLXParams] = None,
    ) -> ExtractionResult:
        """
        Extract SMPL-X parameters from a single image.

        Args:
            image: Input image (HxWx3 uint8 or float32).
            bbox: Optional bounding box (x, y, w, h) for the person.
            prior_params: Optional prior parameters for initialization.

        Returns:
            ExtractionResult with fitted parameters.

        Raises:
            ValueError: If no person is detected.
        """
        import time
        start_time = time.perf_counter()

        self._ensure_model_loaded()

        # Normalize image
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Initialize parameters
        params = prior_params or SMPLXParams()

        # Simulate 2D keypoint detection
        # In a real implementation, this would use a pose detector like OpenPose
        keypoints_2d = self._detect_keypoints(image, bbox)

        if keypoints_2d is None:
            raise ValueError("No person detected in image")

        # Simulate SMPL-X fitting
        # In a real implementation, this would optimize the model parameters
        # to match the detected 2D keypoints
        params = self._fit_model(keypoints_2d, prior_params)

        # Compute mesh vertices if needed
        vertices = None
        faces = None
        if self.config.save_mesh:
            vertices, faces = self._compute_mesh(params)

        # Compute 3D joints
        joints_3d = self._compute_joints_3d(params)

        processing_time = (time.perf_counter() - start_time) * 1000

        return ExtractionResult(
            params=params,
            vertices=vertices,
            faces=faces,
            joints_2d=keypoints_2d,
            joints_3d=joints_3d,
            confidence=0.85,  # Simulated confidence
            processing_time_ms=processing_time,
            metadata={"bbox": bbox, "device": self.device},
        )

    def extract_video(
        self,
        frames: List[npt.NDArray],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[ExtractionResult]:
        """
        Extract SMPL-X parameters from video frames.

        Args:
            frames: List of video frames (HxWx3).
            progress_callback: Optional callback for progress updates.

        Returns:
            List of ExtractionResults, one per frame.
        """
        results = []
        prior_params = None

        for i, frame in enumerate(frames):
            if progress_callback:
                progress = (i / len(frames)) * 100
                progress_callback("extracting", progress)

            try:
                result = self.extract(frame, prior_params=prior_params)
                results.append(result)

                # Use as prior for next frame
                prior_params = result.params

            except ValueError:
                # No detection - carry forward prior
                if prior_params:
                    results.append(ExtractionResult(
                        params=prior_params,
                        confidence=0.0,
                        metadata={"frame": i, "no_detection": True},
                    ))
                else:
                    results.append(ExtractionResult(
                        params=SMPLXParams(),
                        confidence=0.0,
                        metadata={"frame": i, "no_detection": True},
                    ))

        # Apply temporal smoothing if enabled
        if self.config.smooth_poses and len(results) > 1:
            results = self._smooth_results(results)

        if progress_callback:
            progress_callback("complete", 100.0)

        return results

    def _detect_keypoints(
        self,
        image: npt.NDArray[np.float32],
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[npt.NDArray[np.float32]]:
        """
        Detect 2D keypoints in image.

        Args:
            image: Input image (HxWx3 float32).
            bbox: Optional bounding box.

        Returns:
            2D keypoints (N x 3) with x, y, confidence or None if no detection.
        """
        # Simulated keypoint detection
        # In real implementation, this would use OpenPose, MediaPipe, or similar
        h, w = image.shape[:2]

        # Generate plausible keypoints (25 OpenPose keypoints)
        num_keypoints = 25
        keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)

        # Center around image center with some variation
        center_x, center_y = w / 2, h / 2

        # Approximate human proportions
        body_height = min(h, w) * 0.8

        # Head
        keypoints[0] = [center_x, center_y - body_height * 0.4, 0.9]  # Nose

        # Torso
        keypoints[1] = [center_x, center_y - body_height * 0.3, 0.9]  # Neck
        keypoints[8] = [center_x, center_y, 0.9]  # Mid hip

        # Arms (simplified)
        for i, (dx, side) in enumerate([(1, -1), (1, 1)]):  # Left, Right
            keypoints[2 + i * 3] = [center_x + side * body_height * 0.15, center_y - body_height * 0.25, 0.85]  # Shoulder
            keypoints[3 + i * 3] = [center_x + side * body_height * 0.25, center_y - body_height * 0.1, 0.8]  # Elbow
            keypoints[4 + i * 3] = [center_x + side * body_height * 0.3, center_y + body_height * 0.05, 0.75]  # Wrist

        # Legs (simplified)
        for i, side in enumerate([-1, 1]):  # Left, Right
            keypoints[9 + i * 4] = [center_x + side * body_height * 0.1, center_y, 0.85]  # Hip
            keypoints[10 + i * 4] = [center_x + side * body_height * 0.1, center_y + body_height * 0.2, 0.8]  # Knee
            keypoints[11 + i * 4] = [center_x + side * body_height * 0.1, center_y + body_height * 0.4, 0.75]  # Ankle
            keypoints[12 + i * 4] = [center_x + side * body_height * 0.1, center_y + body_height * 0.45, 0.7]  # Foot

        # Check if detection confidence is sufficient
        mean_conf = np.mean(keypoints[:, 2])
        if mean_conf < self.config.min_confidence:
            return None

        return keypoints

    def _fit_model(
        self,
        keypoints_2d: npt.NDArray[np.float32],
        prior_params: Optional[SMPLXParams] = None,
    ) -> SMPLXParams:
        """
        Fit SMPL-X model to 2D keypoints.

        Args:
            keypoints_2d: 2D keypoints (N x 3).
            prior_params: Optional prior parameters.

        Returns:
            Fitted SMPLXParams.
        """
        # Initialize from prior or default
        params = SMPLXParams() if prior_params is None else SMPLXParams(
            body_pose=prior_params.body_pose.copy(),
            left_hand_pose=prior_params.left_hand_pose.copy(),
            right_hand_pose=prior_params.right_hand_pose.copy(),
            betas=prior_params.betas.copy(),
            global_orient=prior_params.global_orient.copy(),
            transl=prior_params.transl.copy(),
        )

        # Simulated optimization
        # In real implementation, this would minimize reprojection error
        # using gradient descent or similar optimization

        # Add small random perturbations to simulate fitting
        noise_scale = 0.1
        params.body_pose += np.random.randn(*params.body_pose.shape).astype(np.float32) * noise_scale

        # Set confidence based on keypoint confidences
        if keypoints_2d is not None:
            params.confidence = np.ones(params.num_body_joints, dtype=np.float32) * np.mean(keypoints_2d[:, 2])

        params.metadata["fitted"] = True
        params.metadata["num_iterations"] = self.config.num_iterations

        return params

    def _compute_mesh(
        self,
        params: SMPLXParams,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
        """
        Compute mesh vertices and faces from SMPL-X parameters.

        Args:
            params: SMPL-X parameters.

        Returns:
            Tuple of (vertices, faces).
        """
        # SMPL-X has approximately 10475 vertices and 20908 faces
        num_vertices = 10475
        num_faces = 20908

        # Generate placeholder mesh
        # In real implementation, this would use the SMPL-X model to compute
        # vertices from pose and shape parameters
        vertices = np.random.randn(num_vertices, 3).astype(np.float32) * 0.5

        # Generate placeholder faces (triangle indices)
        faces = np.random.randint(0, num_vertices, (num_faces, 3), dtype=np.int32)

        return vertices, faces

    def _compute_joints_3d(
        self,
        params: SMPLXParams,
    ) -> npt.NDArray[np.float32]:
        """
        Compute 3D joint positions from SMPL-X parameters.

        Args:
            params: SMPL-X parameters.

        Returns:
            3D joint positions (J x 3).
        """
        # SMPL-X has 55 joints (body + hands + face)
        num_joints = 55

        # Generate placeholder joint positions
        # In real implementation, this would forward kinematics using
        # the SMPL-X skeleton
        joints = np.zeros((num_joints, 3), dtype=np.float32)

        # Set rough positions based on body proportions
        # Pelvis at origin
        joints[0] = params.transl

        # Build kinematic chain (simplified)
        for i in range(1, min(22, num_joints)):
            parent_idx = max(0, i - 1)
            offset = np.array([0, 0.1, 0], dtype=np.float32)  # Simplified offset
            joints[i] = joints[parent_idx] + offset

        return joints

    def _smooth_results(
        self,
        results: List[ExtractionResult],
    ) -> List[ExtractionResult]:
        """
        Apply temporal smoothing to extraction results.

        Args:
            results: List of extraction results.

        Returns:
            Smoothed results.
        """
        window = self.config.temporal_window
        smoothed = []

        for i, result in enumerate(results):
            # Get window of results
            start = max(0, i - window // 2)
            end = min(len(results), i + window // 2 + 1)
            window_results = results[start:end]

            # Average body pose across window
            body_poses = np.stack([r.params.body_pose for r in window_results])
            smoothed_pose = np.mean(body_poses, axis=0)

            # Create smoothed result
            smoothed_params = SMPLXParams(
                body_pose=smoothed_pose.astype(np.float32),
                left_hand_pose=result.params.left_hand_pose,
                right_hand_pose=result.params.right_hand_pose,
                betas=result.params.betas,
                expression=result.params.expression,
                global_orient=result.params.global_orient,
                transl=result.params.transl,
                metadata={"smoothed": True, "window": window},
            )

            smoothed.append(ExtractionResult(
                params=smoothed_params,
                vertices=result.vertices,
                faces=result.faces,
                joints_2d=result.joints_2d,
                joints_3d=result.joints_3d,
                confidence=result.confidence,
                processing_time_ms=result.processing_time_ms,
                metadata={**result.metadata, "smoothed": True},
            ))

        return smoothed

    def extract_from_file(
        self,
        path: Union[str, Path],
    ) -> Union[ExtractionResult, List[ExtractionResult]]:
        """
        Extract SMPL-X parameters from an image or video file.

        Args:
            path: Path to image or video file.

        Returns:
            ExtractionResult for image, or list for video.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        # Image formats
        if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            # Load image (placeholder - real impl would use PIL or OpenCV)
            image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            return self.extract(image)

        # Video formats
        elif suffix in [".mp4", ".avi", ".mov", ".webm"]:
            # Load video frames (placeholder)
            frames = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(30)]
            return self.extract_video(frames)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")


# Convenience functions

def create_t_pose() -> SMPLXParams:
    """
    Create SMPL-X parameters for T-pose.

    Returns:
        SMPLXParams in T-pose.
    """
    params = SMPLXParams()

    # Arms extended horizontally
    # Shoulder joints rotated to extend arms
    params.body_pose[16] = [0, 0, -np.pi / 2]  # Left shoulder
    params.body_pose[17] = [0, 0, np.pi / 2]   # Right shoulder

    params.metadata["pose_name"] = "T-pose"
    return params


def create_a_pose() -> SMPLXParams:
    """
    Create SMPL-X parameters for A-pose.

    Returns:
        SMPLXParams in A-pose.
    """
    params = SMPLXParams()

    # Arms at 45 degrees
    params.body_pose[16] = [0, 0, -np.pi / 4]  # Left shoulder
    params.body_pose[17] = [0, 0, np.pi / 4]   # Right shoulder

    params.metadata["pose_name"] = "A-pose"
    return params


def axis_angle_to_rotation_matrix(axis_angle: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Convert axis-angle representation to rotation matrix.

    Args:
        axis_angle: Axis-angle vector (3,).

    Returns:
        Rotation matrix (3, 3).
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)

    axis = axis_angle / angle

    # Rodrigues' formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float32)

    R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def rotation_matrix_to_axis_angle(R: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Convert rotation matrix to axis-angle representation.

    Args:
        R: Rotation matrix (3, 3).

    Returns:
        Axis-angle vector (3,).
    """
    # Compute angle from trace
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)

    # Compute axis from skew-symmetric part
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float32)

    axis = axis / (2 * np.sin(angle))

    return axis * angle
