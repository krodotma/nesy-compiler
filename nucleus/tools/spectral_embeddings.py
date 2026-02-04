#!/usr/bin/env python3
"""
spectral_embeddings.py - Spectral/Hyperspherical Embeddings

DUALITY-BIND E8: Embed artifacts on S^n with HKS descriptors.

Ring: 1 (Operator)
Protocol: DKIN v29
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class SpectralEmbedding:
    """An embedding on the hypersphere."""
    entity_id: str
    entity_type: str  # "artifact", "trace", "lineage"
    embedding: List[float]
    hks_signature: List[float]  # Heat Kernel Signature
    norm: float = 1.0


class SpectralEmbedder:
    """Embed entities on hypersphere with HKS."""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
    
    def embed_text(self, text: str) -> np.ndarray:
        """Create pseudo-embedding from text hash (placeholder for real embedder)."""
        # In production, use sentence-transformers
        h = hashlib.sha256(text.encode()).digest()
        
        # Expand hash to embedding dimension
        embedding = np.zeros(self.dim)
        for i in range(self.dim):
            embedding[i] = h[i % len(h)] / 255.0 - 0.5
        
        # Project to hypersphere (L2 normalize)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def compute_hks(self, adjacency: np.ndarray, times: List[float] = None) -> np.ndarray:
        """
        Compute Heat Kernel Signature for a graph.
        
        HKS(x, t) = sum_i exp(-lambda_i * t) * phi_i(x)^2
        """
        times = times or [0.1, 1.0, 10.0]
        n = adjacency.shape[0]
        
        # Compute Laplacian
        degree = np.diag(np.sum(adjacency, axis=1))
        laplacian = degree - adjacency
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        except:
            return np.zeros((n, len(times)))
        
        # Compute HKS for each time
        hks = np.zeros((n, len(times)))
        for t_idx, t in enumerate(times):
            for i in range(n):
                hks[:, t_idx] += np.exp(-eigenvalues[i] * t) * (eigenvectors[:, i] ** 2)
        
        return hks
    
    def embed_entity(self, entity_id: str, content: str, entity_type: str = "artifact") -> SpectralEmbedding:
        """Create spectral embedding for an entity."""
        embedding = self.embed_text(content)
        
        # Simple HKS signature based on content structure
        words = content.split()
        n = min(len(words), 10)
        if n > 1:
            # Create simple adjacency (sequential words)
            adj = np.zeros((n, n))
            for i in range(n - 1):
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
            hks = self.compute_hks(adj)
            hks_sig = np.mean(hks, axis=0).tolist()
        else:
            hks_sig = [0.0, 0.0, 0.0]
        
        return SpectralEmbedding(
            entity_id=entity_id,
            entity_type=entity_type,
            embedding=embedding.tolist(),
            hks_signature=hks_sig,
            norm=float(np.linalg.norm(embedding)),
        )
    
    def similarity(self, e1: SpectralEmbedding, e2: SpectralEmbedding) -> float:
        """Compute cosine similarity between embeddings."""
        v1 = np.array(e1.embedding)
        v2 = np.array(e2.embedding)
        return float(np.dot(v1, v2))
    
    def spectral_novelty(self, embedding: SpectralEmbedding, existing: List[SpectralEmbedding]) -> float:
        """Compute how novel an embedding is relative to existing set."""
        if not existing:
            return 1.0
        
        similarities = [self.similarity(embedding, e) for e in existing]
        max_sim = max(similarities)
        return 1.0 - max_sim


def main():
    parser = argparse.ArgumentParser(description="Spectral Embeddings")
    parser.add_argument("text", nargs="?", default="Hello world example text")
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()
    
    embedder = SpectralEmbedder(dim=args.dim)
    
    embedding = embedder.embed_entity("test-entity", args.text)
    
    print(f"Entity: {embedding.entity_id}")
    print(f"Norm: {embedding.norm:.4f} (should be 1.0 for hypersphere)")
    print(f"HKS Signature: {embedding.hks_signature}")
    print(f"Embedding (first 8): {embedding.embedding[:8]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
