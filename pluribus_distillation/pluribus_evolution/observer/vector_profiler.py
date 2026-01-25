#!/usr/bin/env python3
"""
vector_profiler.py - Profiles embedding/manifold space for semantic drift.

Part of the pluribus_evolution observer subsystem.

This observer tracks:
1. Code embedding stability over time
2. Semantic clustering of tools/specs
3. Manifold curvature changes indicating architectural drift
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VectorProfile:
    """Profile of a code artifact in embedding space."""
    path: str
    content_hash: str
    embedding_dim: int = 0  # 0 if not computed
    cluster_id: str | None = None
    semantic_tags: list[str] = field(default_factory=list)


@dataclass
class ManifoldSnapshot:
    """Snapshot of the code manifold at a point in time."""
    timestamp: str
    profiles: list[VectorProfile] = field(default_factory=list)
    cluster_count: int = 0
    centroid_drift: float = 0.0  # Drift from previous snapshot


class VectorProfiler:
    """
    Profiles code in embedding/latent space.

    When a real embedding model is available (e.g., Ollama nomic-embed),
    this profiler can:
    - Compute embeddings for code files
    - Cluster semantically similar tools
    - Track manifold drift over time
    - Identify outliers (potentially misplaced code)

    Without embeddings, falls back to lexical analysis.
    """

    def __init__(self, root_path: str = "/pluribus/nucleus/tools"):
        self.root_path = Path(root_path)
        self.snapshots: list[ManifoldSnapshot] = []

    def profile_file(self, file_path: Path) -> VectorProfile:
        """Create a profile for a single file."""
        import hashlib

        if not file_path.exists():
            return VectorProfile(
                path=str(file_path),
                content_hash="missing"
            )

        content = file_path.read_text(errors="ignore")
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Extract semantic tags from content (lexical fallback)
        tags = []
        if "import json" in content or "json.load" in content:
            tags.append("json")
        if "import asyncio" in content or "async def" in content:
            tags.append("async")
        if "subprocess" in content:
            tags.append("subprocess")
        if "class " in content:
            tags.append("oop")
        if "@dataclass" in content:
            tags.append("dataclass")
        if "def test_" in content or "unittest" in content:
            tags.append("testing")
        if "bus" in content.lower() or "emit" in content.lower():
            tags.append("bus-aware")
        if "click" in content or "argparse" in content:
            tags.append("cli")

        return VectorProfile(
            path=str(file_path),
            content_hash=content_hash,
            semantic_tags=tags
        )

    def profile_directory(self, pattern: str = "**/*.py") -> ManifoldSnapshot:
        """Profile all Python files in directory."""
        profiles = []

        for py_file in self.root_path.glob(pattern):
            if "__pycache__" in str(py_file) or ".venv" in str(py_file):
                continue
            profiles.append(self.profile_file(py_file))

        # Compute clusters based on semantic tags
        tag_groups: dict[frozenset, list[VectorProfile]] = {}
        for p in profiles:
            key = frozenset(p.semantic_tags)
            if key not in tag_groups:
                tag_groups[key] = []
            tag_groups[key].append(p)

        # Assign cluster IDs
        cluster_id = 0
        for key, group in tag_groups.items():
            cluster_name = f"cluster-{cluster_id}"
            for p in group:
                p.cluster_id = cluster_name
            cluster_id += 1

        snapshot = ManifoldSnapshot(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            profiles=profiles,
            cluster_count=len(tag_groups)
        )

        # Calculate drift from previous snapshot
        if self.snapshots:
            prev = self.snapshots[-1]
            prev_hashes = {p.path: p.content_hash for p in prev.profiles}
            changed = sum(
                1 for p in profiles
                if prev_hashes.get(p.path) != p.content_hash
            )
            snapshot.centroid_drift = changed / max(len(profiles), 1)

        self.snapshots.append(snapshot)
        return snapshot

    def find_outliers(self, snapshot: ManifoldSnapshot, threshold: int = 2) -> list[VectorProfile]:
        """Find files with unusual semantic tag combinations."""
        # Files in clusters smaller than threshold
        cluster_sizes: dict[str, int] = {}
        for p in snapshot.profiles:
            if p.cluster_id:
                cluster_sizes[p.cluster_id] = cluster_sizes.get(p.cluster_id, 0) + 1

        outliers = []
        for p in snapshot.profiles:
            if p.cluster_id and cluster_sizes.get(p.cluster_id, 0) < threshold:
                outliers.append(p)

        return outliers

    def to_bus_event(self, snapshot: ManifoldSnapshot) -> dict:
        """Convert snapshot to bus event payload."""
        return {
            "topic": "evolution.observer.manifold",
            "kind": "metric",
            "level": "info",
            "data": {
                "timestamp": snapshot.timestamp,
                "file_count": len(snapshot.profiles),
                "cluster_count": snapshot.cluster_count,
                "centroid_drift": round(snapshot.centroid_drift, 3),
                "top_tags": self._compute_top_tags(snapshot.profiles)
            }
        }

    def _compute_top_tags(self, profiles: list[VectorProfile], n: int = 5) -> dict[str, int]:
        """Count most common semantic tags."""
        tag_counts: dict[str, int] = {}
        for p in profiles:
            for tag in p.semantic_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:n])


if __name__ == "__main__":
    profiler = VectorProfiler()

    print("Profiling nucleus/tools...")
    snapshot = profiler.profile_directory()

    print(f"\nManifold Snapshot ({snapshot.timestamp})")
    print(f"  Files: {len(snapshot.profiles)}")
    print(f"  Clusters: {snapshot.cluster_count}")
    print(f"  Centroid drift: {snapshot.centroid_drift:.3f}")

    outliers = profiler.find_outliers(snapshot)
    if outliers:
        print(f"\nOutliers ({len(outliers)}):")
        for o in outliers[:5]:
            print(f"  {o.path}: {o.semantic_tags}")

    print("\nTop tags:")
    for tag, count in profiler._compute_top_tags(snapshot.profiles).items():
        print(f"  {tag}: {count}")
