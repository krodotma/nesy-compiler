#!/usr/bin/env python3
"""
omega_entelechy.py - Ω-Entelechy Viability Manifold

DUALITY-BIND E11: Population-level slow prior for robust regions.

Ring: 1 (Operator)
Protocol: DKIN v29
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import os

MANIFOLD_DIR = Path(os.environ.get("PLURIBUS_MANIFOLD_DIR", ".pluribus/manifold"))


@dataclass
class ViabilityRegion:
    """A region in the viability manifold."""
    region_id: str
    center: List[float]  # Centroid embedding
    radius: float  # Radius of viability
    population_count: int = 0
    avg_cmp: float = 0.0
    stability: float = 1.0  # How stable this region is


class OmegaEntelechy:
    """
    Ω-Entelechy: Population-level teleology.
    
    Defines a viability manifold where we want lineages to live.
    Distance to manifold becomes a prior for MCTS and bias in CMP.
    """
    
    def __init__(self, manifold_dir: Path = None, dim: int = 32):
        self.manifold_dir = manifold_dir or MANIFOLD_DIR
        self.manifold_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.regions: Dict[str, ViabilityRegion] = {}
    
    def add_region(self, region: ViabilityRegion):
        """Add a viability region."""
        self.regions[region.region_id] = region
        self._save()
    
    def _save(self):
        path = self.manifold_dir / "regions.ndjson"
        with open(path, "w") as f:
            for r in self.regions.values():
                f.write(json.dumps(asdict(r)) + "\n")
    
    def _load(self):
        path = self.manifold_dir / "regions.ndjson"
        if path.exists():
            for line in path.read_text().strip().split("\n"):
                if line:
                    try:
                        data = json.loads(line)
                        r = ViabilityRegion(**data)
                        self.regions[r.region_id] = r
                    except:
                        continue
    
    def distance_to_manifold(self, embedding: List[float]) -> float:
        """
        Compute distance from embedding to nearest viability region.
        
        Lower distance = more viable.
        """
        self._load()
        
        if not self.regions:
            return 0.0  # No manifold defined, everything viable
        
        vec = np.array(embedding)
        min_distance = float("inf")
        
        for region in self.regions.values():
            center = np.array(region.center)
            if len(center) != len(vec):
                # Dimension mismatch, pad or truncate
                if len(center) < len(vec):
                    center = np.pad(center, (0, len(vec) - len(center)))
                else:
                    center = center[:len(vec)]
            
            dist = np.linalg.norm(vec - center) - region.radius
            dist = max(0, dist)  # Inside region = 0 distance
            min_distance = min(min_distance, dist)
        
        return min_distance
    
    def viability_score(self, embedding: List[float]) -> float:
        """
        Compute viability score for an embedding.
        
        Score in [0, 1], higher = more viable.
        """
        distance = self.distance_to_manifold(embedding)
        # Exponential decay with distance
        return math.exp(-distance)
    
    def cmp_bias(self, embedding: List[float], base_cmp: float) -> float:
        """
        Apply viability bias to CMP.
        
        biased_cmp = base_cmp * viability_score
        """
        viability = self.viability_score(embedding)
        return base_cmp * viability
    
    def prune_non_viable(self, lineages: List[Dict], threshold: float = 0.1) -> List[Dict]:
        """Prune lineages outside viability manifold."""
        viable = []
        for lineage in lineages:
            embedding = lineage.get("embedding", [0.0] * self.dim)
            if self.viability_score(embedding) >= threshold:
                viable.append(lineage)
        return viable
    
    def learn_manifold(self, successful_embeddings: List[List[float]], k: int = 3):
        """
        Learn viability regions from successful lineage embeddings.
        Uses simple k-means clustering.
        """
        if not successful_embeddings:
            return
        
        data = np.array(successful_embeddings)
        n = len(data)
        
        # Simple k-means
        k = min(k, n)
        centers = data[np.random.choice(n, k, replace=False)]
        
        for _ in range(10):  # 10 iterations
            # Assign to nearest center
            assignments = []
            for point in data:
                dists = [np.linalg.norm(point - c) for c in centers]
                assignments.append(np.argmin(dists))
            
            # Update centers
            for i in range(k):
                cluster_points = [data[j] for j in range(n) if assignments[j] == i]
                if cluster_points:
                    centers[i] = np.mean(cluster_points, axis=0)
        
        # Create regions
        self.regions = {}
        for i in range(k):
            cluster_points = [data[j] for j in range(n) if assignments[j] == i]
            if cluster_points:
                radius = np.max([np.linalg.norm(p - centers[i]) for p in cluster_points])
                self.regions[f"region-{i}"] = ViabilityRegion(
                    region_id=f"region-{i}",
                    center=centers[i].tolist(),
                    radius=float(radius),
                    population_count=len(cluster_points),
                )
        
        self._save()


def main():
    parser = argparse.ArgumentParser(description="Ω-Entelechy Viability Manifold")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_score = subparsers.add_parser("score", help="Score viability")
    p_score.add_argument("--embedding", type=float, nargs="+", default=[0.5] * 32)
    
    p_add = subparsers.add_parser("add-region", help="Add region")
    p_add.add_argument("--id", required=True)
    p_add.add_argument("--center", type=float, nargs="+", required=True)
    p_add.add_argument("--radius", type=float, default=1.0)
    
    subparsers.add_parser("regions", help="List regions")
    
    args = parser.parse_args()
    entelechy = OmegaEntelechy()
    
    if args.command == "score":
        score = entelechy.viability_score(args.embedding)
        print(f"Viability Score: {score:.3f}")
        return 0
    elif args.command == "add-region":
        region = ViabilityRegion(
            region_id=args.id,
            center=args.center,
            radius=args.radius,
        )
        entelechy.add_region(region)
        print(f"✅ Added region: {args.id}")
        return 0
    elif args.command == "regions":
        entelechy._load()
        for r in entelechy.regions.values():
            print(f"  {r.region_id}: radius={r.radius:.2f}, pop={r.population_count}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
