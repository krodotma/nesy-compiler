"""
VIL ICL+ Geometric Embeddings
Spherical and hyperbolic manifold embeddings for ICL.

Features:
1. Spherical (S^n) embeddings for directional similarity
2. Hyperbolic (H^n) embeddings for hierarchical similarity
3. Riemannian geometry operations
4. Parallel transport between manifolds
5. Exponential/logarithmic maps

Version: 1.0
Date: 2026-01-25
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from nucleus.vil.geometric.metalearning import (
    SphericalManifold,
    HyperbolicManifold,
    PoincareBall,
)


class ManifoldType(str, Enum):
    """Types of manifolds for embeddings."""

    SPHERICAL = "spherical"  # S^n (positive curvature)
    HYPERBOLIC = "hyperbolic"  # H^n (negative curvature)
    EUCLIDEAN = "euclidean"  # R^n (zero curvature)
    PRODUCT = "product"  # S^n x H^n (mixed curvature)


@dataclass
class ManifoldEmbedding:
    """
    Geometric embedding on manifold.

    Contains:
    - Embedding vector
    - Manifold type
    - Curvature parameter
    - Metadata
    """

    vector: np.ndarray
    manifold_type: ManifoldType
    curvature: float = 1.0  # K > 0: spherical, K < 0: hyperbolic
    dimension: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dimension == 0:
            self.dimension = len(self.vector)
        if self.metadata is None:
            self.metadata = {}


class GeometricEmbedder:
    """
    Geometric embedder for ICL examples.

    Features:
    1. Spherical embedding (S^n)
    2. Hyperbolic embedding (Poincaré ball H^n)
    3. Manifold distance computation
    4. Parallel transport
    5. Product space embeddings
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        default_manifold: ManifoldType = ManifoldType.SPHERICAL,
        curvature: float = 1.0,
    ):
        self.embedding_dim = embedding_dim
        self.default_manifold = default_manifold
        self.curvature = curvature

        # Manifold instances
        self.spherical = SphericalManifold()
        self.hyperbolic = PoincareBall(epsilon=0.1)

    def embed_spherical(
        self,
        data: np.ndarray,
        normalize: bool = True,
    ) -> ManifoldEmbedding:
        """
        Embed data onto spherical manifold S^n.

        Args:
            data: Input vector
            normalize: Whether to L2 normalize

        Returns:
            ManifoldEmbedding on S^n
        """
        if normalize:
            # Project onto unit sphere
            embedded = self.spherical.project_to_sphere(data)
        else:
            embedded = data.copy()

        return ManifoldEmbedding(
            vector=embedded,
            manifold_type=ManifoldType.SPHERICAL,
            curvature=1.0,  # Positive curvature
            dimension=len(embedded),
        )

    def embed_hyperbolic(
        self,
        data: np.ndarray,
        epsilon: float = 0.1,
    ) -> ManifoldEmbedding:
        """
        Embed data onto hyperbolic manifold H^n (Poincaré ball).

        Args:
            data: Input vector
            epsilon: Poincaré ball radius

        Returns:
            ManifoldEmbedding on H^n
        """
        # Project onto Poincaré ball
        embedded = self.hyperbolic.project(data, epsilon=epsilon)

        return ManifoldEmbedding(
            vector=embedded,
            manifold_type=ManifoldType.HYPERBOLIC,
            curvature=-1.0,  # Negative curvature
            dimension=len(embedded),
            metadata={"epsilon": epsilon},
        )

    def embed_product(
        self,
        data: np.ndarray,
        split_ratio: float = 0.5,
    ) -> ManifoldEmbedding:
        """
        Embed data onto product manifold S^n x H^n.

        Splits embedding into spherical and hyperbolic components.

        Args:
            data: Input vector
            split_ratio: Ratio of spherical component

        Returns:
            ManifoldEmbedding on S^n x H^n
        """
        dim = len(data)
        spherical_dim = int(dim * split_ratio)
        hyperbolic_dim = dim - spherical_dim

        # Split data
        spherical_data = data[:spherical_dim]
        hyperbolic_data = data[spherical_dim:]

        # Embed each component
        spherical_emb = self.embed_spherical(spherical_data)
        hyperbolic_emb = self.embed_hyperbolic(hyperbolic_data)

        # Concatenate
        combined = np.concatenate([spherical_emb.vector, hyperbolic_emb.vector])

        return ManifoldEmbedding(
            vector=combined,
            manifold_type=ManifoldType.PRODUCT,
            curvature=0.0,  # Mixed curvature
            dimension=len(combined),
            metadata={
                "spherical_dim": spherical_dim,
                "hyperbolic_dim": hyperbolic_dim,
                "split_ratio": split_ratio,
            },
        )

    def compute_distance(
        self,
        emb1: ManifoldEmbedding,
        emb2: ManifoldEmbedding,
    ) -> float:
        """
        Compute geodesic distance between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Geodesic distance
        """
        if emb1.manifold_type != emb2.manifold_type:
            raise ValueError(f"Cannot compare {emb1.manifold_type} and {emb2.manifold_type}")

        if emb1.manifold_type == ManifoldType.SPHERICAL:
            # Great circle distance
            return self.spherical.great_circle_distance(emb1.vector, emb2.vector)

        elif emb1.manifold_type == ManifoldType.HYPERBOLIC:
            # Poincaré distance
            return self.hyperbolic.poincare_distance(emb1.vector, emb2.vector)

        elif emb1.manifold_type == ManifoldType.EUCLIDEAN:
            # Euclidean distance
            return np.linalg.norm(emb1.vector - emb2.vector)

        elif emb1.manifold_type == ManifoldType.PRODUCT:
            # Split and compute distance for each component
            split = emb1.metadata.get("spherical_dim", len(emb1.vector) // 2)

            spherical_dist = self.spherical.great_circle_distance(
                emb1.vector[:split],
                emb2.vector[:split],
            )
            hyperbolic_dist = self.hyperbolic.poincare_distance(
                emb1.vector[split:],
                emb2.vector[split:],
            )

            # Weighted combination
            return spherical_dist + hyperbolic_dist

        else:
            return np.linalg.norm(emb1.vector - emb2.vector)

    def parallel_transport(
        self,
        vector: np.ndarray,
        from_point: ManifoldEmbedding,
        to_point: ManifoldEmbedding,
    ) -> np.ndarray:
        """
        Parallel transport vector between manifold points.

        Args:
            vector: Vector to transport
            from_point: Source point
            to_point: Target point

        Returns:
            Transported vector
        """
        if from_point.manifold_type == ManifoldType.SPHERICAL:
            return self.spherical.parallel_transport(vector, from_point.vector, to_point.vector)

        elif from_point.manifold_type == ManifoldType.HYPERBOLIC:
            return self.hyperbolic.parallel_transport(vector, from_point.vector, to_point.vector)

        else:
            # No transport for Euclidean
            return vector.copy()

    def exponential_map(
        self,
        point: ManifoldEmbedding,
        tangent_vector: np.ndarray,
    ) -> ManifoldEmbedding:
        """
        Exponential map: move from point along tangent vector.

        Args:
            point: Starting point
            tangent_vector: Direction and magnitude

        Returns:
            New point on manifold
        """
        if point.manifold_type == ManifoldType.SPHERICAL:
            # Exponential map on sphere
            new_vector = self.spherical.project_to_sphere(
                point.vector + tangent_vector * 0.1  # Small step
            )
            return ManifoldEmbedding(
                vector=new_vector,
                manifold_type=point.manifold_type,
                curvature=point.curvature,
            )

        elif point.manifold_type == ManifoldType.HYPERBOLIC:
            # Exponential map on hyperbolic
            # Approximation: project and scale
            new_vector = self.hyperbolic.project(
                point.vector + tangent_vector * 0.1,
                epsilon=point.metadata.get("epsilon", 0.1),
            )
            return ManifoldEmbedding(
                vector=new_vector,
                manifold_type=point.manifold_type,
                curvature=point.curvature,
                metadata=point.metadata,
            )

        else:
            # Euclidean: simple addition
            new_vector = point.vector + tangent_vector
            return ManifoldEmbedding(
                vector=new_vector,
                manifold_type=point.manifold_type,
                curvature=0.0,
            )

    def logarithmic_map(
        self,
        from_point: ManifoldEmbedding,
        to_point: ManifoldEmbedding,
    ) -> np.ndarray:
        """
        Logarithmic map: compute tangent vector from point to point.

        Args:
            from_point: Starting point
            to_point: Target point

        Returns:
            Tangent vector
        """
        # Simplified: direction toward target
        direction = to_point.vector - from_point.vector

        # Scale by geodesic distance
        distance = self.compute_distance(from_point, to_point)
        if distance > 0:
            direction = direction / np.linalg.norm(direction) * distance

        return direction

    def interpolate(
        self,
        emb1: ManifoldEmbedding,
        emb2: ManifoldEmbedding,
        t: float = 0.5,
    ) -> ManifoldEmbedding:
        """
        Geodesic interpolation between embeddings.

        Args:
            emb1: Start embedding
            emb2: End embedding
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated embedding
        """
        # Get tangent vector
        tangent = self.logarithmic_map(emb1, emb2)

        # Scale by t
        scaled_tangent = tangent * t

        # Exponential map
        return self.exponential_map(emb1, scaled_tangent)

    def compute_curvature(
        self,
        embeddings: List[ManifoldEmbedding],
    ) -> float:
        """
        Estimate local curvature from embeddings.

        Args:
            embeddings: List of nearby embeddings

        Returns:
            Estimated curvature
        """
        if len(embeddings) < 3:
            return 0.0

        # Compute pairwise distances
        distances = []
        for i, emb1 in enumerate(embeddings):
            for emb2 in embeddings[i+1:]:
                distances.append(self.compute_distance(emb1, emb2))

        # Use triangle inequality deviation to estimate curvature
        # Positive curvature: triangles "bulge" outward
        # Negative curvature: triangles "thin" inward
        if len(distances) < 3:
            return 0.0

        # Simplified estimation
        avg_distance = np.mean(distances)
        variance = np.var(distances)

        # High variance suggests curvature
        return float(variance / (avg_distance ** 2 + 1e-8))


def create_geometric_embedder(
    embedding_dim: int = 64,
    default_manifold: ManifoldType = ManifoldType.SPHERICAL,
    curvature: float = 1.0,
) -> GeometricEmbedder:
    """Create geometric embedder with default config."""
    return GeometricEmbedder(
        embedding_dim=embedding_dim,
        default_manifold=default_manifold,
        curvature=curvature,
    )


__all__ = [
    "ManifoldType",
    "ManifoldEmbedding",
    "GeometricEmbedder",
    "create_geometric_embedder",
]
