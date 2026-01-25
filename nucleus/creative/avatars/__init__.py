"""
Avatars Subsystem
=================

3DGS, Neural PBR, SMPL-X, procedural parameters.

Modules:
- smplx_extractor: SMPL-X body model extraction from images/video
- gaussian_splatting: 3D Gaussian Splatting representation
- deformable: Deformable gaussian avatars with skinning
- neural_pbr: Neural physically-based rendering materials
- procedural: Procedural avatar parameter generation
- bus_events: Bus event definitions for avatars subsystem
"""

from __future__ import annotations

# Import submodules
from . import smplx_extractor
from . import gaussian_splatting
from . import deformable
from . import neural_pbr
from . import procedural
from . import bus_events

# Export from smplx_extractor
from .smplx_extractor import (
    SMPLXParams,
    SMPLXExtractor,
    ExtractionConfig,
    ExtractionResult,
    BodyPart,
    HandJoint,
    create_t_pose,
    create_a_pose,
    axis_angle_to_rotation_matrix,
    rotation_matrix_to_axis_angle,
)

# Export from gaussian_splatting
from .gaussian_splatting import (
    Gaussian3D,
    GaussianSplatCloud,
    GaussianCloudOperations,
    SHDegree,
)

# Export from deformable
from .deformable import (
    DeformableGaussians,
    DeformationResult,
    DeformationType,
    BindingType,
    SkinningWeights,
    DeformableGaussiansConfig,
)

# Export from neural_pbr
from .neural_pbr import (
    NeuralPBRMaterial,
    MaterialProperties,
    MaterialType,
    TextureType,
    PBRTexture,
    NeuralMaterialEncoder,
    create_skin_material,
    create_hair_material,
    create_cloth_material,
)

# Export from procedural
from .procedural import (
    ProceduralAvatarGenerator,
    ProceduralAvatarParams,
    ProceduralBodyParams,
    ProceduralFaceParams,
    ProceduralClothingParams,
    ProceduralHairParams,
    AvatarStyle,
    Gender,
    BodyType,
    FaceType,
    ClothingType,
    HairStyle,
    generate_random_avatar,
    generate_avatar_from_description,
)

# Export from bus_events
from .bus_events import (
    AvatarBusEvent,
    AvatarEventType,
    AvatarBusEmitter,
    AVATAR_TOPICS,
    create_extraction_started_event,
    create_extraction_progress_event,
    create_extraction_completed_event,
    create_extraction_failed_event,
    create_gaussian_cloud_event,
    create_deformation_started_event,
    create_deformation_completed_event,
    create_material_event,
    create_procedural_started_event,
    create_procedural_completed_event,
    create_render_completed_event,
    get_emitter,
    emit_event,
)

__all__ = [
    # Submodules
    "smplx_extractor",
    "gaussian_splatting",
    "deformable",
    "neural_pbr",
    "procedural",
    "bus_events",

    # SMPL-X
    "SMPLXParams",
    "SMPLXExtractor",
    "ExtractionConfig",
    "ExtractionResult",
    "BodyPart",
    "HandJoint",
    "create_t_pose",
    "create_a_pose",
    "axis_angle_to_rotation_matrix",
    "rotation_matrix_to_axis_angle",

    # Gaussian Splatting
    "Gaussian3D",
    "GaussianSplatCloud",
    "GaussianCloudOperations",
    "SHDegree",

    # Deformable
    "DeformableGaussians",
    "DeformationResult",
    "DeformationType",
    "BindingType",
    "SkinningWeights",
    "DeformableGaussiansConfig",

    # Neural PBR
    "NeuralPBRMaterial",
    "MaterialProperties",
    "MaterialType",
    "TextureType",
    "PBRTexture",
    "NeuralMaterialEncoder",
    "create_skin_material",
    "create_hair_material",
    "create_cloth_material",

    # Procedural
    "ProceduralAvatarGenerator",
    "ProceduralAvatarParams",
    "ProceduralBodyParams",
    "ProceduralFaceParams",
    "ProceduralClothingParams",
    "ProceduralHairParams",
    "AvatarStyle",
    "Gender",
    "BodyType",
    "FaceType",
    "ClothingType",
    "HairStyle",
    "generate_random_avatar",
    "generate_avatar_from_description",

    # Bus Events
    "AvatarBusEvent",
    "AvatarEventType",
    "AvatarBusEmitter",
    "AVATAR_TOPICS",
    "create_extraction_started_event",
    "create_extraction_progress_event",
    "create_extraction_completed_event",
    "create_extraction_failed_event",
    "create_gaussian_cloud_event",
    "create_deformation_started_event",
    "create_deformation_completed_event",
    "create_material_event",
    "create_procedural_started_event",
    "create_procedural_completed_event",
    "create_render_completed_event",
    "get_emitter",
    "emit_event",
]
