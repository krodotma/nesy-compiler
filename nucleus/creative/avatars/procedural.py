"""
Procedural Avatar Generation
============================

Procedural generation of avatar parameters including body shape,
facial features, clothing, and accessories. Uses parametric models
combined with controlled randomization for diverse avatar creation.

Features:
- Body shape parameter generation (SMPL-X betas)
- Facial feature synthesis
- Procedural clothing and texture generation
- Accessory placement
- Style-guided generation
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
import hashlib

from .smplx_extractor import SMPLXParams


class Gender(Enum):
    """Gender for body shape generation."""
    MALE = auto()
    FEMALE = auto()
    NEUTRAL = auto()


class BodyType(Enum):
    """Body type presets."""
    AVERAGE = auto()
    ATHLETIC = auto()
    SLIM = auto()
    MUSCULAR = auto()
    HEAVY = auto()
    TALL = auto()
    SHORT = auto()


class FaceType(Enum):
    """Face shape presets."""
    OVAL = auto()
    ROUND = auto()
    SQUARE = auto()
    HEART = auto()
    OBLONG = auto()
    DIAMOND = auto()


class ClothingType(Enum):
    """Types of clothing."""
    CASUAL = auto()
    FORMAL = auto()
    ATHLETIC = auto()
    BUSINESS = auto()
    CREATIVE = auto()
    MINIMAL = auto()


class HairStyle(Enum):
    """Hair style presets."""
    SHORT = auto()
    MEDIUM = auto()
    LONG = auto()
    BALD = auto()
    PONYTAIL = auto()
    BRAIDS = auto()
    CURLY = auto()
    WAVY = auto()


@dataclass
class AvatarStyle:
    """
    Style configuration for avatar generation.

    Defines high-level aesthetic choices that influence
    procedural generation.

    Attributes:
        gender: Gender for body shape.
        body_type: Body type preset.
        face_type: Face shape preset.
        age_range: Age range (min, max).
        ethnicity_hints: Hints for ethnic features.
        clothing_type: Clothing style.
        hair_style: Hair style.
        color_palette: Preferred colors.
        style_seed: Random seed for reproducibility.
    """
    gender: Gender = Gender.NEUTRAL
    body_type: BodyType = BodyType.AVERAGE
    face_type: FaceType = FaceType.OVAL
    age_range: Tuple[int, int] = (20, 40)
    ethnicity_hints: List[str] = field(default_factory=list)
    clothing_type: ClothingType = ClothingType.CASUAL
    hair_style: HairStyle = HairStyle.MEDIUM
    color_palette: List[str] = field(default_factory=list)
    style_seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize random seed if not provided."""
        if self.style_seed is None:
            self.style_seed = np.random.randint(0, 2**31)

    def get_rng(self) -> np.random.Generator:
        """Get seeded random number generator."""
        return np.random.default_rng(self.style_seed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gender": self.gender.name,
            "body_type": self.body_type.name,
            "face_type": self.face_type.name,
            "age_range": self.age_range,
            "ethnicity_hints": self.ethnicity_hints,
            "clothing_type": self.clothing_type.name,
            "hair_style": self.hair_style.name,
            "color_palette": self.color_palette,
            "style_seed": self.style_seed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AvatarStyle":
        """Create from dictionary."""
        return cls(
            gender=Gender[data.get("gender", "NEUTRAL")],
            body_type=BodyType[data.get("body_type", "AVERAGE")],
            face_type=FaceType[data.get("face_type", "OVAL")],
            age_range=tuple(data.get("age_range", (20, 40))),
            ethnicity_hints=data.get("ethnicity_hints", []),
            clothing_type=ClothingType[data.get("clothing_type", "CASUAL")],
            hair_style=HairStyle[data.get("hair_style", "MEDIUM")],
            color_palette=data.get("color_palette", []),
            style_seed=data.get("style_seed"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProceduralBodyParams:
    """
    Procedurally generated body parameters.

    Attributes:
        betas: SMPL-X shape coefficients.
        height_cm: Target height in centimeters.
        weight_kg: Target weight in kilograms.
        proportions: Body proportions dictionary.
        muscle_definition: Muscle definition level [0, 1].
        body_fat_percentage: Body fat percentage.
    """
    betas: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(10, dtype=np.float32)
    )
    height_cm: float = 170.0
    weight_kg: float = 70.0
    proportions: Dict[str, float] = field(default_factory=dict)
    muscle_definition: float = 0.5
    body_fat_percentage: float = 20.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_smplx_params(self) -> SMPLXParams:
        """Convert to SMPL-X parameters."""
        return SMPLXParams(
            betas=self.betas.copy(),
            metadata={
                "procedural": True,
                "height_cm": self.height_cm,
                "weight_kg": self.weight_kg,
            },
        )


@dataclass
class ProceduralFaceParams:
    """
    Procedurally generated face parameters.

    Attributes:
        expression_coeffs: FLAME expression coefficients.
        jaw_params: Jaw shape parameters.
        face_shape: Face shape parameters.
        eye_params: Eye shape and position parameters.
        nose_params: Nose shape parameters.
        mouth_params: Mouth shape parameters.
        eyebrow_params: Eyebrow shape parameters.
    """
    expression_coeffs: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(10, dtype=np.float32)
    )
    jaw_params: Dict[str, float] = field(default_factory=dict)
    face_shape: Dict[str, float] = field(default_factory=dict)
    eye_params: Dict[str, float] = field(default_factory=dict)
    nose_params: Dict[str, float] = field(default_factory=dict)
    mouth_params: Dict[str, float] = field(default_factory=dict)
    eyebrow_params: Dict[str, float] = field(default_factory=dict)
    skin_params: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProceduralClothingParams:
    """
    Procedurally generated clothing parameters.

    Attributes:
        garment_types: List of garment types.
        colors: Garment colors.
        patterns: Pattern types.
        fit: Fit level (tight to loose).
        layers: Number of clothing layers.
        accessories: List of accessories.
    """
    garment_types: List[str] = field(default_factory=list)
    colors: List[npt.NDArray[np.float32]] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    fit: float = 0.5
    layers: int = 1
    accessories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProceduralHairParams:
    """
    Procedurally generated hair parameters.

    Attributes:
        style: Hair style enum.
        length: Hair length (0-1).
        density: Hair density (0-1).
        color: Hair color (RGB).
        highlight_color: Highlight color (optional).
        curliness: Curl intensity (0-1).
        thickness: Strand thickness (0-1).
    """
    style: HairStyle = HairStyle.MEDIUM
    length: float = 0.5
    density: float = 0.8
    color: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.array([0.2, 0.15, 0.1], dtype=np.float32)
    )
    highlight_color: Optional[npt.NDArray[np.float32]] = None
    curliness: float = 0.0
    thickness: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProceduralAvatarParams:
    """
    Complete procedural avatar parameters.

    Combines body, face, clothing, and hair parameters
    for full avatar generation.

    Attributes:
        id: Unique avatar identifier.
        style: Style configuration.
        body: Body parameters.
        face: Face parameters.
        clothing: Clothing parameters.
        hair: Hair parameters.
        created_at: Creation timestamp.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    style: AvatarStyle = field(default_factory=AvatarStyle)
    body: ProceduralBodyParams = field(default_factory=ProceduralBodyParams)
    face: ProceduralFaceParams = field(default_factory=ProceduralFaceParams)
    clothing: ProceduralClothingParams = field(default_factory=ProceduralClothingParams)
    hair: ProceduralHairParams = field(default_factory=ProceduralHairParams)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "style": self.style.to_dict(),
            "body": {
                "betas": self.body.betas.tolist(),
                "height_cm": self.body.height_cm,
                "weight_kg": self.body.weight_kg,
                "proportions": self.body.proportions,
                "muscle_definition": self.body.muscle_definition,
                "body_fat_percentage": self.body.body_fat_percentage,
            },
            "face": {
                "expression_coeffs": self.face.expression_coeffs.tolist(),
                "jaw_params": self.face.jaw_params,
                "face_shape": self.face.face_shape,
                "eye_params": self.face.eye_params,
                "nose_params": self.face.nose_params,
                "mouth_params": self.face.mouth_params,
            },
            "clothing": {
                "garment_types": self.clothing.garment_types,
                "patterns": self.clothing.patterns,
                "fit": self.clothing.fit,
                "layers": self.clothing.layers,
                "accessories": self.clothing.accessories,
            },
            "hair": {
                "style": self.hair.style.name,
                "length": self.hair.length,
                "density": self.hair.density,
                "color": self.hair.color.tolist(),
                "curliness": self.hair.curliness,
                "thickness": self.hair.thickness,
            },
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProceduralAvatarParams":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        style = AvatarStyle.from_dict(data.get("style", {}))

        body_data = data.get("body", {})
        body = ProceduralBodyParams(
            betas=np.array(body_data.get("betas", np.zeros(10))),
            height_cm=body_data.get("height_cm", 170.0),
            weight_kg=body_data.get("weight_kg", 70.0),
            proportions=body_data.get("proportions", {}),
            muscle_definition=body_data.get("muscle_definition", 0.5),
            body_fat_percentage=body_data.get("body_fat_percentage", 20.0),
        )

        face_data = data.get("face", {})
        face = ProceduralFaceParams(
            expression_coeffs=np.array(face_data.get("expression_coeffs", np.zeros(10))),
            jaw_params=face_data.get("jaw_params", {}),
            face_shape=face_data.get("face_shape", {}),
            eye_params=face_data.get("eye_params", {}),
            nose_params=face_data.get("nose_params", {}),
            mouth_params=face_data.get("mouth_params", {}),
        )

        clothing_data = data.get("clothing", {})
        clothing = ProceduralClothingParams(
            garment_types=clothing_data.get("garment_types", []),
            patterns=clothing_data.get("patterns", []),
            fit=clothing_data.get("fit", 0.5),
            layers=clothing_data.get("layers", 1),
            accessories=clothing_data.get("accessories", []),
        )

        hair_data = data.get("hair", {})
        hair = ProceduralHairParams(
            style=HairStyle[hair_data.get("style", "MEDIUM")],
            length=hair_data.get("length", 0.5),
            density=hair_data.get("density", 0.8),
            color=np.array(hair_data.get("color", [0.2, 0.15, 0.1])),
            curliness=hair_data.get("curliness", 0.0),
            thickness=hair_data.get("thickness", 0.5),
        )

        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            style=style,
            body=body,
            face=face,
            clothing=clothing,
            hair=hair,
            metadata=data.get("metadata", {}),
        )


class ProceduralAvatarGenerator:
    """
    Generator for procedural avatar parameters.

    Provides methods to generate diverse avatar configurations
    based on style presets and controlled randomization.

    Example:
        >>> generator = ProceduralAvatarGenerator()
        >>> style = AvatarStyle(gender=Gender.FEMALE, body_type=BodyType.ATHLETIC)
        >>> params = generator.generate(style)
        >>> smplx = params.body.to_smplx_params()
    """

    def __init__(
        self,
        default_num_betas: int = 10,
        default_num_expression: int = 10,
    ):
        """
        Initialize generator.

        Args:
            default_num_betas: Default number of shape coefficients.
            default_num_expression: Default number of expression coefficients.
        """
        self.num_betas = default_num_betas
        self.num_expression = default_num_expression

        # Body type parameter distributions
        self._body_type_params = {
            BodyType.AVERAGE: {"height": (165, 180), "weight": (60, 80), "betas_scale": 0.5},
            BodyType.ATHLETIC: {"height": (170, 185), "weight": (65, 85), "betas_scale": 0.8, "muscle": 0.7},
            BodyType.SLIM: {"height": (165, 180), "weight": (50, 65), "betas_scale": 0.6},
            BodyType.MUSCULAR: {"height": (170, 190), "weight": (80, 100), "betas_scale": 1.0, "muscle": 0.9},
            BodyType.HEAVY: {"height": (160, 180), "weight": (85, 110), "betas_scale": 0.9},
            BodyType.TALL: {"height": (180, 200), "weight": (70, 90), "betas_scale": 0.7},
            BodyType.SHORT: {"height": (150, 165), "weight": (50, 70), "betas_scale": 0.6},
        }

        # Gender-specific adjustments
        self._gender_adjustments = {
            Gender.MALE: {"height_offset": 10, "weight_offset": 10, "betas_offset": np.array([0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
            Gender.FEMALE: {"height_offset": -5, "weight_offset": -8, "betas_offset": np.array([-0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
            Gender.NEUTRAL: {"height_offset": 0, "weight_offset": 0, "betas_offset": np.zeros(10)},
        }

        # Hair color presets
        self._hair_colors = {
            "black": np.array([0.05, 0.04, 0.03]),
            "dark_brown": np.array([0.15, 0.10, 0.05]),
            "brown": np.array([0.3, 0.2, 0.1]),
            "light_brown": np.array([0.5, 0.35, 0.2]),
            "blonde": np.array([0.8, 0.7, 0.4]),
            "red": np.array([0.6, 0.2, 0.1]),
            "auburn": np.array([0.5, 0.25, 0.15]),
            "gray": np.array([0.5, 0.5, 0.5]),
            "white": np.array([0.9, 0.9, 0.85]),
        }

        # Clothing type configurations
        self._clothing_configs = {
            ClothingType.CASUAL: {
                "garments": ["t-shirt", "jeans", "sneakers"],
                "patterns": ["solid", "stripes", "graphic"],
                "fit_range": (0.4, 0.6),
            },
            ClothingType.FORMAL: {
                "garments": ["dress_shirt", "suit_jacket", "dress_pants", "dress_shoes"],
                "patterns": ["solid", "pinstripe", "subtle_check"],
                "fit_range": (0.3, 0.5),
            },
            ClothingType.ATHLETIC: {
                "garments": ["athletic_top", "shorts", "athletic_shoes"],
                "patterns": ["solid", "color_block"],
                "fit_range": (0.2, 0.4),
            },
            ClothingType.BUSINESS: {
                "garments": ["blouse", "blazer", "slacks", "loafers"],
                "patterns": ["solid", "subtle_pattern"],
                "fit_range": (0.4, 0.5),
            },
            ClothingType.CREATIVE: {
                "garments": ["unique_top", "artistic_pants", "creative_shoes"],
                "patterns": ["abstract", "artistic", "bold"],
                "fit_range": (0.3, 0.7),
            },
            ClothingType.MINIMAL: {
                "garments": ["simple_top", "simple_pants"],
                "patterns": ["solid"],
                "fit_range": (0.4, 0.5),
            },
        }

    def generate(
        self,
        style: Optional[AvatarStyle] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ProceduralAvatarParams:
        """
        Generate procedural avatar parameters.

        Args:
            style: Style configuration (uses defaults if None).
            progress_callback: Optional progress callback.

        Returns:
            Complete procedural avatar parameters.
        """
        if style is None:
            style = AvatarStyle()

        rng = style.get_rng()

        if progress_callback:
            progress_callback("generating_body", 0.0)

        # Generate body parameters
        body = self._generate_body(style, rng)

        if progress_callback:
            progress_callback("generating_face", 25.0)

        # Generate face parameters
        face = self._generate_face(style, rng)

        if progress_callback:
            progress_callback("generating_clothing", 50.0)

        # Generate clothing parameters
        clothing = self._generate_clothing(style, rng)

        if progress_callback:
            progress_callback("generating_hair", 75.0)

        # Generate hair parameters
        hair = self._generate_hair(style, rng)

        if progress_callback:
            progress_callback("complete", 100.0)

        return ProceduralAvatarParams(
            style=style,
            body=body,
            face=face,
            clothing=clothing,
            hair=hair,
            metadata={
                "generator_version": "1.0",
                "seed": style.style_seed,
            },
        )

    def _generate_body(
        self,
        style: AvatarStyle,
        rng: np.random.Generator,
    ) -> ProceduralBodyParams:
        """Generate body parameters."""
        # Get body type parameters
        body_params = self._body_type_params[style.body_type]
        gender_adj = self._gender_adjustments[style.gender]

        # Generate height and weight
        height_range = body_params["height"]
        weight_range = body_params["weight"]

        height = rng.uniform(
            height_range[0] + gender_adj["height_offset"],
            height_range[1] + gender_adj["height_offset"]
        )
        weight = rng.uniform(
            weight_range[0] + gender_adj["weight_offset"],
            weight_range[1] + gender_adj["weight_offset"]
        )

        # Generate SMPL-X betas
        betas_scale = body_params.get("betas_scale", 0.5)
        betas = rng.standard_normal(self.num_betas).astype(np.float32) * betas_scale

        # Apply gender offset
        betas[:len(gender_adj["betas_offset"])] += gender_adj["betas_offset"][:self.num_betas]

        # Muscle definition
        muscle = body_params.get("muscle", 0.5)
        muscle_variation = rng.uniform(-0.2, 0.2)
        muscle_definition = np.clip(muscle + muscle_variation, 0, 1)

        # Body fat (inversely correlated with muscle for athletes)
        base_fat = 20.0
        if style.body_type == BodyType.ATHLETIC:
            base_fat = 15.0
        elif style.body_type == BodyType.HEAVY:
            base_fat = 30.0
        body_fat = base_fat + rng.uniform(-5, 5)

        # Proportions
        proportions = {
            "torso_ratio": float(rng.uniform(0.45, 0.55)),
            "leg_ratio": float(rng.uniform(0.45, 0.55)),
            "shoulder_width": float(rng.uniform(0.4, 0.6)),
            "hip_width": float(rng.uniform(0.4, 0.6)),
        }

        return ProceduralBodyParams(
            betas=betas,
            height_cm=height,
            weight_kg=weight,
            proportions=proportions,
            muscle_definition=muscle_definition,
            body_fat_percentage=body_fat,
        )

    def _generate_face(
        self,
        style: AvatarStyle,
        rng: np.random.Generator,
    ) -> ProceduralFaceParams:
        """Generate face parameters."""
        # Face shape based on type
        face_shapes = {
            FaceType.OVAL: {"jaw_width": 0.4, "cheek_width": 0.5, "forehead_width": 0.5},
            FaceType.ROUND: {"jaw_width": 0.5, "cheek_width": 0.6, "forehead_width": 0.5},
            FaceType.SQUARE: {"jaw_width": 0.6, "cheek_width": 0.5, "forehead_width": 0.6},
            FaceType.HEART: {"jaw_width": 0.3, "cheek_width": 0.5, "forehead_width": 0.6},
            FaceType.OBLONG: {"jaw_width": 0.4, "cheek_width": 0.4, "forehead_width": 0.4},
            FaceType.DIAMOND: {"jaw_width": 0.3, "cheek_width": 0.6, "forehead_width": 0.4},
        }

        base_shape = face_shapes[style.face_type]
        face_shape = {
            k: float(v + rng.uniform(-0.1, 0.1))
            for k, v in base_shape.items()
        }

        # Expression coefficients (neutral with slight variations)
        expression_coeffs = rng.standard_normal(self.num_expression).astype(np.float32) * 0.1

        # Eye parameters
        eye_params = {
            "size": float(rng.uniform(0.4, 0.6)),
            "spacing": float(rng.uniform(0.4, 0.6)),
            "openness": float(rng.uniform(0.4, 0.6)),
            "shape": float(rng.uniform(0.3, 0.7)),  # 0=round, 1=almond
        }

        # Nose parameters
        nose_params = {
            "length": float(rng.uniform(0.4, 0.6)),
            "width": float(rng.uniform(0.4, 0.6)),
            "bridge_height": float(rng.uniform(0.4, 0.6)),
            "tip_shape": float(rng.uniform(0.3, 0.7)),
        }

        # Mouth parameters
        mouth_params = {
            "width": float(rng.uniform(0.4, 0.6)),
            "fullness_upper": float(rng.uniform(0.3, 0.7)),
            "fullness_lower": float(rng.uniform(0.3, 0.7)),
        }

        # Eyebrow parameters
        eyebrow_params = {
            "thickness": float(rng.uniform(0.3, 0.7)),
            "arch": float(rng.uniform(0.3, 0.7)),
            "spacing": float(rng.uniform(0.4, 0.6)),
        }

        # Skin parameters (affected by age)
        min_age, max_age = style.age_range
        age = rng.uniform(min_age, max_age)
        age_factor = (age - 20) / 60  # Normalize 20-80 to 0-1

        skin_params = {
            "age": float(age),
            "wrinkle_intensity": float(max(0, age_factor * 0.5 + rng.uniform(-0.1, 0.1))),
            "skin_tone": float(rng.uniform(0.2, 0.8)),
            "freckles": float(rng.uniform(0, 0.3)),
        }

        return ProceduralFaceParams(
            expression_coeffs=expression_coeffs,
            face_shape=face_shape,
            eye_params=eye_params,
            nose_params=nose_params,
            mouth_params=mouth_params,
            eyebrow_params=eyebrow_params,
            skin_params=skin_params,
        )

    def _generate_clothing(
        self,
        style: AvatarStyle,
        rng: np.random.Generator,
    ) -> ProceduralClothingParams:
        """Generate clothing parameters."""
        config = self._clothing_configs[style.clothing_type]

        # Select garments
        garments = config["garments"].copy()

        # Select patterns
        num_patterns = len(garments)
        patterns = [rng.choice(config["patterns"]) for _ in range(num_patterns)]

        # Generate colors
        colors = []
        if style.color_palette:
            # Use provided palette
            for _ in range(num_patterns):
                hex_color = rng.choice(style.color_palette)
                colors.append(self._hex_to_rgb(hex_color))
        else:
            # Generate harmonious colors
            base_hue = rng.uniform(0, 1)
            for i in range(num_patterns):
                hue = (base_hue + i * 0.15) % 1.0
                sat = rng.uniform(0.3, 0.7)
                val = rng.uniform(0.4, 0.9)
                colors.append(self._hsv_to_rgb(hue, sat, val))

        # Fit
        fit_range = config["fit_range"]
        fit = rng.uniform(fit_range[0], fit_range[1])

        # Accessories
        possible_accessories = ["watch", "necklace", "bracelet", "ring", "earrings", "glasses"]
        num_accessories = rng.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        accessories = list(rng.choice(possible_accessories, size=num_accessories, replace=False))

        return ProceduralClothingParams(
            garment_types=garments,
            colors=colors,
            patterns=patterns,
            fit=fit,
            layers=len(garments) // 2 + 1,
            accessories=accessories,
        )

    def _generate_hair(
        self,
        style: AvatarStyle,
        rng: np.random.Generator,
    ) -> ProceduralHairParams:
        """Generate hair parameters."""
        # Length based on style
        length_map = {
            HairStyle.SHORT: (0.1, 0.3),
            HairStyle.MEDIUM: (0.3, 0.5),
            HairStyle.LONG: (0.6, 0.9),
            HairStyle.BALD: (0.0, 0.05),
            HairStyle.PONYTAIL: (0.5, 0.8),
            HairStyle.BRAIDS: (0.5, 0.9),
            HairStyle.CURLY: (0.2, 0.6),
            HairStyle.WAVY: (0.3, 0.7),
        }

        length_range = length_map[style.hair_style]
        length = rng.uniform(length_range[0], length_range[1])

        # Hair color
        color_name = rng.choice(list(self._hair_colors.keys()))
        base_color = self._hair_colors[color_name].copy()

        # Add variation
        color_variation = rng.uniform(-0.05, 0.05, size=3)
        color = np.clip(base_color + color_variation, 0, 1).astype(np.float32)

        # Highlight color (sometimes)
        highlight_color = None
        if rng.random() < 0.3:
            highlight_base = rng.choice(list(self._hair_colors.keys()))
            highlight_color = self._hair_colors[highlight_base].copy().astype(np.float32)

        # Curliness
        curliness = 0.0
        if style.hair_style in [HairStyle.CURLY]:
            curliness = rng.uniform(0.6, 1.0)
        elif style.hair_style == HairStyle.WAVY:
            curliness = rng.uniform(0.3, 0.5)
        else:
            curliness = rng.uniform(0, 0.2)

        return ProceduralHairParams(
            style=style.hair_style,
            length=length,
            density=rng.uniform(0.6, 1.0),
            color=color,
            highlight_color=highlight_color,
            curliness=curliness,
            thickness=rng.uniform(0.3, 0.7),
        )

    def _hex_to_rgb(self, hex_color: str) -> npt.NDArray[np.float32]:
        """Convert hex color to RGB array."""
        hex_color = hex_color.lstrip("#")
        return np.array([
            int(hex_color[0:2], 16) / 255,
            int(hex_color[2:4], 16) / 255,
            int(hex_color[4:6], 16) / 255,
        ], dtype=np.float32)

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> npt.NDArray[np.float32]:
        """Convert HSV to RGB."""
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c

        if h < 1/6:
            rgb = (c, x, 0)
        elif h < 2/6:
            rgb = (x, c, 0)
        elif h < 3/6:
            rgb = (0, c, x)
        elif h < 4/6:
            rgb = (0, x, c)
        elif h < 5/6:
            rgb = (x, 0, c)
        else:
            rgb = (c, 0, x)

        return np.array([rgb[0] + m, rgb[1] + m, rgb[2] + m], dtype=np.float32)

    def generate_batch(
        self,
        count: int,
        base_style: Optional[AvatarStyle] = None,
        variation_level: float = 0.5,
    ) -> List[ProceduralAvatarParams]:
        """
        Generate a batch of varied avatar parameters.

        Args:
            count: Number of avatars to generate.
            base_style: Base style to vary from.
            variation_level: How much variation (0=identical, 1=very different).

        Returns:
            List of avatar parameters.
        """
        if base_style is None:
            base_style = AvatarStyle()

        avatars = []
        base_seed = base_style.style_seed or 42

        for i in range(count):
            # Create varied style
            varied_style = AvatarStyle(
                gender=base_style.gender,
                body_type=base_style.body_type,
                face_type=base_style.face_type,
                age_range=base_style.age_range,
                ethnicity_hints=base_style.ethnicity_hints,
                clothing_type=base_style.clothing_type,
                hair_style=base_style.hair_style,
                color_palette=base_style.color_palette,
                style_seed=base_seed + i,  # Different seed for each
            )

            avatar = self.generate(varied_style)
            avatars.append(avatar)

        return avatars


def generate_random_avatar(seed: Optional[int] = None) -> ProceduralAvatarParams:
    """
    Quick function to generate a completely random avatar.

    Args:
        seed: Optional random seed.

    Returns:
        Random avatar parameters.
    """
    if seed is None:
        seed = np.random.randint(0, 2**31)

    rng = np.random.default_rng(seed)

    style = AvatarStyle(
        gender=rng.choice(list(Gender)),
        body_type=rng.choice(list(BodyType)),
        face_type=rng.choice(list(FaceType)),
        age_range=(20, 60),
        clothing_type=rng.choice(list(ClothingType)),
        hair_style=rng.choice(list(HairStyle)),
        style_seed=seed,
    )

    generator = ProceduralAvatarGenerator()
    return generator.generate(style)


def generate_avatar_from_description(description: str) -> ProceduralAvatarParams:
    """
    Generate avatar from text description (simplified).

    Args:
        description: Text description of desired avatar.

    Returns:
        Avatar parameters matching description.
    """
    # Create deterministic seed from description
    desc_hash = hashlib.md5(description.encode()).hexdigest()
    seed = int(desc_hash[:8], 16)

    # Parse description for hints
    desc_lower = description.lower()

    # Gender detection
    if any(w in desc_lower for w in ["male", "man", "boy", "guy"]):
        gender = Gender.MALE
    elif any(w in desc_lower for w in ["female", "woman", "girl", "lady"]):
        gender = Gender.FEMALE
    else:
        gender = Gender.NEUTRAL

    # Body type detection
    body_type = BodyType.AVERAGE
    if any(w in desc_lower for w in ["athletic", "fit", "sporty"]):
        body_type = BodyType.ATHLETIC
    elif any(w in desc_lower for w in ["slim", "thin", "slender"]):
        body_type = BodyType.SLIM
    elif any(w in desc_lower for w in ["muscular", "buff", "strong"]):
        body_type = BodyType.MUSCULAR
    elif any(w in desc_lower for w in ["heavy", "large", "big"]):
        body_type = BodyType.HEAVY
    elif "tall" in desc_lower:
        body_type = BodyType.TALL
    elif "short" in desc_lower:
        body_type = BodyType.SHORT

    # Hair style detection
    hair_style = HairStyle.MEDIUM
    if "bald" in desc_lower:
        hair_style = HairStyle.BALD
    elif any(w in desc_lower for w in ["long hair", "long-haired"]):
        hair_style = HairStyle.LONG
    elif any(w in desc_lower for w in ["short hair", "short-haired"]):
        hair_style = HairStyle.SHORT
    elif "curly" in desc_lower:
        hair_style = HairStyle.CURLY
    elif "wavy" in desc_lower:
        hair_style = HairStyle.WAVY
    elif "ponytail" in desc_lower:
        hair_style = HairStyle.PONYTAIL
    elif "braids" in desc_lower:
        hair_style = HairStyle.BRAIDS

    # Clothing detection
    clothing_type = ClothingType.CASUAL
    if any(w in desc_lower for w in ["formal", "suit", "elegant"]):
        clothing_type = ClothingType.FORMAL
    elif any(w in desc_lower for w in ["athletic", "sporty", "gym"]):
        clothing_type = ClothingType.ATHLETIC
    elif any(w in desc_lower for w in ["business", "professional", "office"]):
        clothing_type = ClothingType.BUSINESS
    elif any(w in desc_lower for w in ["creative", "artistic", "unique"]):
        clothing_type = ClothingType.CREATIVE

    # Age detection
    age_range = (25, 35)
    if any(w in desc_lower for w in ["young", "teenager", "teen"]):
        age_range = (15, 25)
    elif any(w in desc_lower for w in ["middle-aged", "middle aged"]):
        age_range = (40, 55)
    elif any(w in desc_lower for w in ["elderly", "old", "senior"]):
        age_range = (60, 80)

    style = AvatarStyle(
        gender=gender,
        body_type=body_type,
        face_type=FaceType.OVAL,
        age_range=age_range,
        clothing_type=clothing_type,
        hair_style=hair_style,
        style_seed=seed,
        metadata={"description": description},
    )

    generator = ProceduralAvatarGenerator()
    return generator.generate(style)
