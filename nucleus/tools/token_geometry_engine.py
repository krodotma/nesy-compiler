#!/usr/bin/env python3
"""
Token Geometry Engine - Unified Lexer/Parser with Vec2Vec and N-Sphere Geometry

Combines:
- tiktoken BPE tokenization for accurate token counting
- AUOM (Atomic Units of Meaning) semantic tokenization
- Sextet encoding (6-channel multimodal representation)
- Vec2Vec object transformation
- N-sphere hyperspherical embeddings
- Superpositional LTL validation

Architecture:
    Raw Text → BPE Tokens → AUOM Units → Sextet Vectors → N-Sphere Projection
                                ↓
                        Vec2Vec Transform
                                ↓
                        Knowledge Graph
                                ↓
                        LTL Validation
                                ↓
                        Bus Events

Usage:
    from token_geometry_engine import TokenGeometryEngine

    engine = TokenGeometryEngine()
    result = engine.process("Check the system status")
    # Returns: TokenGeometryResult with tokens, vectors, auom_units, ltl_valid
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, Literal, Sequence

import numpy as np

# Optional: tiktoken for accurate BPE tokenization
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Optional: sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Default embedding dimension (must match model output)
DEFAULT_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
NSPHERE_DIM = 128  # Projected n-sphere dimension

# Sextet channels (6-tuple multimodal representation)
class SextetChannel(Enum):
    SEMANTIC = 0      # Meaning/intent
    SYNTACTIC = 1     # Grammatical structure
    PRAGMATIC = 2     # Context/usage
    TEMPORAL = 3      # Time/sequence
    MODAL = 4         # Modality (text/audio/visual/code)
    AFFECTIVE = 5     # Sentiment/emotion

# AUOM categories (Atomic Units of Meaning)
class AUOMCategory(Enum):
    ENTITY = "entity"           # Named entities, objects
    ACTION = "action"           # Verbs, operations
    RELATION = "relation"       # Connections, predicates
    MODIFIER = "modifier"       # Adjectives, adverbs
    QUANTIFIER = "quantifier"   # Numbers, amounts
    TEMPORAL = "temporal"       # Time expressions
    SPATIAL = "spatial"         # Location expressions
    OPERATOR = "operator"       # Semops operators (CKIN, ITERATE, etc.)
    CONNECTOR = "connector"     # Logical connectives

# LTL operators for temporal logic
class LTLOperator(Enum):
    ALWAYS = "□"      # G (globally)
    EVENTUALLY = "◇"  # F (finally)
    NEXT = "○"        # X (next)
    UNTIL = "U"       # U (until)
    RELEASE = "R"     # R (release)
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"

# Semops operator patterns for AUOM detection
SEMOPS_PATTERNS = {
    r'\b(ckin|dkin|chkin|checking\s+in)\b': 'CKIN',
    r'\b(iterate|oiterate)\b': 'ITERATE',
    r'\b(realagents?|dispatch)\b': 'REALAGENTS',
    r'\b(pbflush|flush)\b': 'PBFLUSH',
    r'\b(pbdeep|deep\s+analysis)\b': 'PBDEEP',
    r'\b(pbassimilate|pbassimilation)\b': 'PBASSIMILATE',
    r'\b(pluribus)\b': 'PLURIBUS',
    r'\b(beam|log\s+entry)\b': 'BEAM',
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BPEToken:
    """Single BPE token with metadata."""
    id: int
    text: str
    position: int
    byte_offset: int

@dataclass
class AUOMUnit:
    """Atomic Unit of Meaning - semantic token."""
    id: str
    text: str
    category: AUOMCategory
    confidence: float
    span: tuple[int, int]  # (start, end) character positions
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SextetVector:
    """6-channel multimodal vector representation."""
    semantic: np.ndarray      # Channel 0
    syntactic: np.ndarray     # Channel 1
    pragmatic: np.ndarray     # Channel 2
    temporal: np.ndarray      # Channel 3
    modal: np.ndarray         # Channel 4
    affective: np.ndarray     # Channel 5

    def to_numpy(self) -> np.ndarray:
        """Concatenate all channels into single vector."""
        return np.concatenate([
            self.semantic, self.syntactic, self.pragmatic,
            self.temporal, self.modal, self.affective
        ])

    def channel(self, ch: SextetChannel) -> np.ndarray:
        """Get specific channel by enum."""
        return [self.semantic, self.syntactic, self.pragmatic,
                self.temporal, self.modal, self.affective][ch.value]

@dataclass
class NSpherePoint:
    """Point on n-dimensional hypersphere."""
    coords: np.ndarray  # Cartesian coordinates
    radius: float = 1.0

    @property
    def angular(self) -> np.ndarray:
        """Convert to angular coordinates (n-1 angles)."""
        n = len(self.coords)
        angles = np.zeros(n - 1)
        for i in range(n - 1):
            denom = np.sqrt(np.sum(self.coords[i:]**2))
            if denom > 1e-10:
                angles[i] = np.arccos(np.clip(self.coords[i] / denom, -1, 1))
        return angles

    def geodesic_distance(self, other: 'NSpherePoint') -> float:
        """Compute geodesic (great circle) distance to another point."""
        dot = np.clip(np.dot(self.coords, other.coords) / (self.radius * other.radius), -1, 1)
        return np.arccos(dot) * self.radius

@dataclass
class LTLFormula:
    """Linear Temporal Logic formula."""
    operator: LTLOperator | None
    operands: list['LTLFormula | str']

    def __str__(self) -> str:
        if self.operator is None:
            return str(self.operands[0]) if self.operands else "⊤"
        if len(self.operands) == 1:
            return f"{self.operator.value}{self.operands[0]}"
        return f"({self.operands[0]} {self.operator.value} {self.operands[1]})"

    @classmethod
    def always(cls, prop: str) -> 'LTLFormula':
        """□p - p holds in all future states."""
        return cls(LTLOperator.ALWAYS, [prop])

    @classmethod
    def eventually(cls, prop: str) -> 'LTLFormula':
        """◇p - p holds in some future state."""
        return cls(LTLOperator.EVENTUALLY, [prop])

    @classmethod
    def until(cls, p: str, q: str) -> 'LTLFormula':
        """p U q - p holds until q becomes true."""
        return cls(LTLOperator.UNTIL, [p, q])

@dataclass
class SuperpositionalState:
    """Quantum-inspired superpositional state for parallel possibilities."""
    basis_states: list[str]
    amplitudes: np.ndarray  # Complex amplitudes

    def __post_init__(self):
        # Normalize amplitudes
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 1e-10:
            self.amplitudes = self.amplitudes / norm

    def probability(self, state: str) -> float:
        """Get probability of observing specific state."""
        if state not in self.basis_states:
            return 0.0
        idx = self.basis_states.index(state)
        return float(np.abs(self.amplitudes[idx])**2)

    def collapse(self, rng: np.random.Generator | None = None) -> str:
        """Collapse superposition to single state (measurement)."""
        rng = rng or np.random.default_rng()
        probs = np.abs(self.amplitudes)**2
        return rng.choice(self.basis_states, p=probs)

    def interfere(self, other: 'SuperpositionalState') -> 'SuperpositionalState':
        """Quantum interference between two superpositional states."""
        # Combine basis states
        all_states = list(set(self.basis_states) | set(other.basis_states))
        new_amps = np.zeros(len(all_states), dtype=complex)

        for i, state in enumerate(all_states):
            if state in self.basis_states:
                new_amps[i] += self.amplitudes[self.basis_states.index(state)]
            if state in other.basis_states:
                new_amps[i] += other.amplitudes[other.basis_states.index(state)]

        return SuperpositionalState(all_states, new_amps)

@dataclass
class Vec2VecTransform:
    """Vector-to-vector transformation specification."""
    source_space: str
    target_space: str
    transform_matrix: np.ndarray | None = None
    nonlinear_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def apply(self, vec: np.ndarray) -> np.ndarray:
        """Apply transformation to vector."""
        result = vec
        if self.transform_matrix is not None:
            result = self.transform_matrix @ result
        if self.nonlinear_fn is not None:
            result = self.nonlinear_fn(result)
        return result

@dataclass
class TokenGeometryResult:
    """Complete result from token geometry processing."""
    # Input
    raw_text: str

    # Tokenization
    bpe_tokens: list[BPEToken]
    token_count: int

    # Semantic analysis
    auom_units: list[AUOMUnit]
    detected_operators: list[str]

    # Vector representations
    sextet_vector: SextetVector | None
    nsphere_point: NSpherePoint | None

    # Temporal logic
    ltl_formula: LTLFormula | None
    ltl_valid: bool
    superposition: SuperpositionalState | None

    # Metadata
    processing_time_ms: float
    engine_version: str = "1.0.0"

# =============================================================================
# TOKENIZER BACKENDS
# =============================================================================

class TiktokenBackend:
    """tiktoken BPE tokenizer backend."""

    def __init__(self, encoding: str = "cl100k_base"):
        if not TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken not installed: pip install tiktoken")
        self.encoding_name = encoding
        self.enc = tiktoken.get_encoding(encoding)

    def encode(self, text: str) -> list[BPEToken]:
        """Encode text to BPE tokens."""
        token_ids = self.enc.encode(text)
        tokens = []
        byte_offset = 0

        for i, tid in enumerate(token_ids):
            token_bytes = self.enc.decode_single_token_bytes(tid)
            token_text = token_bytes.decode('utf-8', errors='replace')
            tokens.append(BPEToken(
                id=tid,
                text=token_text,
                position=i,
                byte_offset=byte_offset
            ))
            byte_offset += len(token_bytes)

        return tokens

    def decode(self, tokens: list[BPEToken]) -> str:
        """Decode BPE tokens to text."""
        return self.enc.decode([t.id for t in tokens])

    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.enc.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens."""
        tokens = self.enc.encode(text)[:max_tokens]
        return self.enc.decode(tokens)

class FallbackTokenizer:
    """Fallback tokenizer when tiktoken unavailable."""

    def __init__(self):
        self.word_pattern = re.compile(r'\S+|\s+')

    def encode(self, text: str) -> list[BPEToken]:
        """Simple whitespace-aware tokenization."""
        tokens = []
        offset = 0
        for i, match in enumerate(self.word_pattern.finditer(text)):
            tokens.append(BPEToken(
                id=hash(match.group()) % (2**31),
                text=match.group(),
                position=i,
                byte_offset=offset
            ))
            offset += len(match.group().encode('utf-8'))
        return tokens

    def decode(self, tokens: list[BPEToken]) -> str:
        return ''.join(t.text for t in tokens)

    def count(self, text: str) -> int:
        # Approximation: ~4 chars per token
        return max(1, len(text) // 4)

    def truncate(self, text: str, max_tokens: int) -> str:
        approx_chars = max_tokens * 4
        return text[:approx_chars]

# =============================================================================
# EMBEDDING BACKENDS
# =============================================================================

class SentenceTransformerBackend:
    """Sentence-BERT embedding backend."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed single text."""
        return self.model.encode([text], convert_to_numpy=True)[0]

class HashEmbedding:
    """Deterministic hash-based embedding (fallback)."""

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed_single(t) for t in texts])

    def embed_single(self, text: str) -> np.ndarray:
        """Hash text to pseudo-embedding vector."""
        # Use SHA256 and expand to desired dimension
        h = hashlib.sha256(text.encode('utf-8')).digest()
        # Expand hash to full dimension using SHAKE
        shake = hashlib.shake_256(text.encode('utf-8'))
        raw = shake.digest(self.dim * 4)
        vec = np.frombuffer(raw, dtype=np.float32)[:self.dim]
        # Normalize to unit sphere
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        return vec

# =============================================================================
# AUOM ANALYZER
# =============================================================================

class AUOMAnalyzer:
    """Atomic Units of Meaning analyzer."""

    def __init__(self):
        self.semops_re = {
            re.compile(pattern, re.IGNORECASE): op
            for pattern, op in SEMOPS_PATTERNS.items()
        }

        # Simple POS-like patterns for AUOM categorization
        self.category_patterns = {
            AUOMCategory.TEMPORAL: re.compile(
                r'\b(now|today|tomorrow|yesterday|always|never|'
                r'before|after|during|since|until|when|while|'
                r'\d{4}-\d{2}-\d{2}|\d+\s*(ms|sec|min|hour|day)s?)\b',
                re.IGNORECASE
            ),
            AUOMCategory.QUANTIFIER: re.compile(
                r'\b(\d+|one|two|three|four|five|ten|hundred|'
                r'all|some|none|many|few|most|every)\b',
                re.IGNORECASE
            ),
            AUOMCategory.SPATIAL: re.compile(
                r'\b(here|there|where|above|below|inside|outside|'
                r'near|far|between|through|across)\b',
                re.IGNORECASE
            ),
            AUOMCategory.CONNECTOR: re.compile(
                r'\b(and|or|but|if|then|because|therefore|'
                r'however|although|unless|while)\b',
                re.IGNORECASE
            ),
        }

    def analyze(self, text: str, embedder: Any = None) -> list[AUOMUnit]:
        """Extract AUOM units from text."""
        units = []

        # Detect semops operators
        for pattern, op_name in self.semops_re.items():
            for match in pattern.finditer(text):
                units.append(AUOMUnit(
                    id=f"auom-{uuid.uuid4().hex[:8]}",
                    text=match.group(),
                    category=AUOMCategory.OPERATOR,
                    confidence=0.95,
                    span=(match.start(), match.end()),
                    metadata={"operator": op_name}
                ))

        # Detect other categories
        for category, pattern in self.category_patterns.items():
            for match in pattern.finditer(text):
                # Skip if already captured as operator
                if any(u.span[0] <= match.start() < u.span[1] for u in units):
                    continue
                units.append(AUOMUnit(
                    id=f"auom-{uuid.uuid4().hex[:8]}",
                    text=match.group(),
                    category=category,
                    confidence=0.8,
                    span=(match.start(), match.end()),
                ))

        # Add embeddings if embedder available
        if embedder is not None:
            texts = [u.text for u in units]
            if texts:
                embeddings = embedder.embed(texts)
                for i, unit in enumerate(units):
                    unit.embedding = embeddings[i]

        # Sort by position
        units.sort(key=lambda u: u.span[0])
        return units

# =============================================================================
# N-SPHERE GEOMETRY
# =============================================================================

class NSphereGeometry:
    """Hyperspherical geometry for embedding space."""

    def __init__(self, dim: int = NSPHERE_DIM):
        self.dim = dim
        self.rng = np.random.default_rng()

    def project_to_sphere(self, vec: np.ndarray) -> NSpherePoint:
        """Project arbitrary vector onto unit n-sphere."""
        # Reduce dimension if needed
        if len(vec) > self.dim:
            # Use random projection (Johnson-Lindenstrauss)
            proj_matrix = self.rng.standard_normal((self.dim, len(vec))) / np.sqrt(self.dim)
            vec = proj_matrix @ vec
        elif len(vec) < self.dim:
            # Pad with zeros
            vec = np.pad(vec, (0, self.dim - len(vec)))

        # Normalize to unit sphere
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        else:
            vec = np.zeros(self.dim)
            vec[0] = 1.0  # Default to north pole

        return NSpherePoint(coords=vec, radius=1.0)

    def geodesic_interpolate(
        self,
        p1: NSpherePoint,
        p2: NSpherePoint,
        t: float
    ) -> NSpherePoint:
        """Spherical linear interpolation (SLERP)."""
        dot = np.clip(np.dot(p1.coords, p2.coords), -1, 1)
        theta = np.arccos(dot)

        if theta < 1e-10:
            return p1

        sin_theta = np.sin(theta)
        a = np.sin((1 - t) * theta) / sin_theta
        b = np.sin(t * theta) / sin_theta

        coords = a * p1.coords + b * p2.coords
        return NSpherePoint(coords=coords, radius=1.0)

    def cluster_geodesic(
        self,
        points: list[NSpherePoint],
        n_clusters: int = 5
    ) -> list[list[int]]:
        """Cluster points on n-sphere using geodesic k-means."""
        if len(points) < n_clusters:
            return [[i] for i in range(len(points))]

        # Initialize centroids randomly
        centroid_indices = self.rng.choice(len(points), n_clusters, replace=False)
        centroids = [points[i] for i in centroid_indices]

        for _ in range(20):  # Max iterations
            # Assign points to nearest centroid
            clusters = [[] for _ in range(n_clusters)]
            for i, point in enumerate(points):
                distances = [point.geodesic_distance(c) for c in centroids]
                clusters[np.argmin(distances)].append(i)

            # Update centroids (Fréchet mean on sphere)
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    mean_coords = np.mean([points[i].coords for i in cluster], axis=0)
                    norm = np.linalg.norm(mean_coords)
                    if norm > 1e-10:
                        mean_coords = mean_coords / norm
                    new_centroids.append(NSpherePoint(coords=mean_coords))
                else:
                    new_centroids.append(centroids[len(new_centroids)])

            centroids = new_centroids

        return clusters

    def exponential_map(self, base: NSpherePoint, tangent: np.ndarray) -> NSpherePoint:
        """Exponential map from tangent space to sphere."""
        norm = np.linalg.norm(tangent)
        if norm < 1e-10:
            return base

        direction = tangent / norm
        coords = np.cos(norm) * base.coords + np.sin(norm) * direction
        return NSpherePoint(coords=coords, radius=base.radius)

    def logarithmic_map(self, base: NSpherePoint, target: NSpherePoint) -> np.ndarray:
        """Logarithmic map from sphere to tangent space at base."""
        dot = np.clip(np.dot(base.coords, target.coords), -1, 1)
        theta = np.arccos(dot)

        if theta < 1e-10:
            return np.zeros_like(base.coords)

        direction = target.coords - dot * base.coords
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm

        return theta * direction

# =============================================================================
# SEXTET ENCODER
# =============================================================================

class SextetEncoder:
    """6-channel multimodal encoder."""

    def __init__(self, channel_dim: int = 64):
        self.channel_dim = channel_dim
        self.hash_embed = HashEmbedding(channel_dim)

    def encode(
        self,
        text: str,
        auom_units: list[AUOMUnit] | None = None,
        embedder: Any = None
    ) -> SextetVector:
        """Encode text into sextet vector."""

        # Channel 0: Semantic (base embedding)
        if embedder is not None:
            semantic = embedder.embed_single(text)
            if len(semantic) > self.channel_dim:
                semantic = semantic[:self.channel_dim]
            elif len(semantic) < self.channel_dim:
                semantic = np.pad(semantic, (0, self.channel_dim - len(semantic)))
        else:
            semantic = self.hash_embed.embed_single(text)

        # Channel 1: Syntactic (structure-based)
        syntactic = self._encode_syntactic(text)

        # Channel 2: Pragmatic (context/usage)
        pragmatic = self._encode_pragmatic(text, auom_units)

        # Channel 3: Temporal (time expressions)
        temporal = self._encode_temporal(text, auom_units)

        # Channel 4: Modal (modality indicators)
        modal = self._encode_modal(text)

        # Channel 5: Affective (sentiment)
        affective = self._encode_affective(text)

        return SextetVector(
            semantic=semantic,
            syntactic=syntactic,
            pragmatic=pragmatic,
            temporal=temporal,
            modal=modal,
            affective=affective
        )

    def _encode_syntactic(self, text: str) -> np.ndarray:
        """Encode syntactic structure."""
        vec = np.zeros(self.channel_dim)

        # Simple syntactic features
        features = {
            'word_count': len(text.split()),
            'char_count': len(text),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'question': 1.0 if '?' in text else 0.0,
            'imperative': 1.0 if text.strip().endswith('!') or text[0].isupper() else 0.0,
            'parenthetical': text.count('(') + text.count('['),
            'punctuation_density': sum(1 for c in text if c in '.,;:!?') / max(1, len(text)),
        }

        for i, (_, v) in enumerate(features.items()):
            if i < self.channel_dim:
                vec[i] = float(v)

        # Hash rest for deterministic fill
        hash_part = self.hash_embed.embed_single(f"syn:{text}")
        vec[len(features):] = hash_part[len(features):]

        return vec / (np.linalg.norm(vec) + 1e-10)

    def _encode_pragmatic(self, text: str, auom_units: list[AUOMUnit] | None) -> np.ndarray:
        """Encode pragmatic/contextual features."""
        vec = np.zeros(self.channel_dim)

        # Count AUOM categories
        if auom_units:
            category_counts = {}
            for unit in auom_units:
                cat = unit.category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1

            for i, cat in enumerate(AUOMCategory):
                if i < self.channel_dim:
                    vec[i] = category_counts.get(cat.value, 0)

        # Add hash component
        hash_part = self.hash_embed.embed_single(f"prag:{text}")
        offset = len(AUOMCategory)
        vec[offset:] = hash_part[offset:]

        return vec / (np.linalg.norm(vec) + 1e-10)

    def _encode_temporal(self, text: str, auom_units: list[AUOMUnit] | None) -> np.ndarray:
        """Encode temporal features."""
        vec = np.zeros(self.channel_dim)

        # Temporal indicators
        temporal_words = ['now', 'then', 'before', 'after', 'always', 'never',
                         'today', 'tomorrow', 'yesterday', 'soon', 'later']

        text_lower = text.lower()
        for i, word in enumerate(temporal_words):
            if i < self.channel_dim and word in text_lower:
                vec[i] = 1.0

        # Count temporal AUOM units
        if auom_units:
            temporal_count = sum(1 for u in auom_units if u.category == AUOMCategory.TEMPORAL)
            vec[min(len(temporal_words), self.channel_dim - 1)] = temporal_count

        # Fill rest with hash
        hash_part = self.hash_embed.embed_single(f"temp:{text}")
        offset = len(temporal_words) + 1
        vec[offset:] = hash_part[offset:]

        return vec / (np.linalg.norm(vec) + 1e-10)

    def _encode_modal(self, text: str) -> np.ndarray:
        """Encode modality indicators."""
        vec = np.zeros(self.channel_dim)

        # Modality detection heuristics
        modalities = {
            'code': bool(re.search(r'```|def |class |function |import |const |let |var ', text)),
            'url': bool(re.search(r'https?://', text)),
            'path': bool(re.search(r'/\w+/\w+|\\w+\\w+', text)),
            'json': bool(re.search(r'\{["\']?\w+["\']?\s*:', text)),
            'math': bool(re.search(r'[+\-*/=<>]{2,}|\d+\.\d+', text)),
            'list': bool(re.search(r'^\s*[-*]\s', text, re.MULTILINE)),
        }

        for i, (_, present) in enumerate(modalities.items()):
            if i < self.channel_dim:
                vec[i] = 1.0 if present else 0.0

        hash_part = self.hash_embed.embed_single(f"modal:{text}")
        vec[len(modalities):] = hash_part[len(modalities):]

        return vec / (np.linalg.norm(vec) + 1e-10)

    def _encode_affective(self, text: str) -> np.ndarray:
        """Encode affective/sentiment features."""
        vec = np.zeros(self.channel_dim)

        # Simple sentiment lexicon
        positive = ['good', 'great', 'excellent', 'success', 'happy', 'wonderful', 'amazing', 'perfect']
        negative = ['bad', 'error', 'fail', 'wrong', 'problem', 'issue', 'broken', 'critical']

        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)

        vec[0] = pos_count
        vec[1] = neg_count
        vec[2] = pos_count - neg_count  # Valence
        vec[3] = 1.0 if '!' in text else 0.0  # Intensity
        vec[4] = text.count('!') + text.count('?')  # Arousal proxy

        hash_part = self.hash_embed.embed_single(f"affect:{text}")
        vec[5:] = hash_part[5:]

        return vec / (np.linalg.norm(vec) + 1e-10)

# =============================================================================
# SUPERPOSITIONAL LTL VALIDATOR
# =============================================================================

class SuperpositionalLTL:
    """Superpositional Linear Temporal Logic validator."""

    def __init__(self):
        self.state_history: list[SuperpositionalState] = []

    def create_superposition(self, states: list[str], weights: list[float] | None = None) -> SuperpositionalState:
        """Create superposition from basis states."""
        if weights is None:
            weights = [1.0] * len(states)
        amplitudes = np.array(weights, dtype=complex)
        return SuperpositionalState(states, amplitudes)

    def validate_always(self, prop: str, states: list[SuperpositionalState]) -> tuple[bool, float]:
        """Validate □p (always p) over state sequence."""
        if not states:
            return True, 1.0

        probs = [s.probability(prop) for s in states]
        min_prob = min(probs) if probs else 0.0

        # In superpositional logic, "always" means high probability in all states
        return min_prob > 0.5, min_prob

    def validate_eventually(self, prop: str, states: list[SuperpositionalState]) -> tuple[bool, float]:
        """Validate ◇p (eventually p) over state sequence."""
        if not states:
            return False, 0.0

        probs = [s.probability(prop) for s in states]
        max_prob = max(probs) if probs else 0.0

        return max_prob > 0.5, max_prob

    def validate_until(
        self,
        p: str,
        q: str,
        states: list[SuperpositionalState]
    ) -> tuple[bool, float]:
        """Validate p U q (p until q) over state sequence."""
        if not states:
            return False, 0.0

        q_found = False
        p_held = True
        confidence = 1.0

        for state in states:
            q_prob = state.probability(q)
            p_prob = state.probability(p)

            if q_prob > 0.5:
                q_found = True
                confidence = min(confidence, q_prob)
                break

            if p_prob <= 0.5:
                p_held = False
                confidence = min(confidence, p_prob)
                break

            confidence = min(confidence, p_prob)

        return q_found and p_held, confidence

    def validate_formula(
        self,
        formula: LTLFormula,
        states: list[SuperpositionalState]
    ) -> tuple[bool, float]:
        """Validate arbitrary LTL formula."""
        if formula.operator is None:
            # Atomic proposition
            prop = str(formula.operands[0])
            if states:
                prob = states[-1].probability(prop)
                return prob > 0.5, prob
            return False, 0.0

        if formula.operator == LTLOperator.ALWAYS:
            return self.validate_always(str(formula.operands[0]), states)

        if formula.operator == LTLOperator.EVENTUALLY:
            return self.validate_eventually(str(formula.operands[0]), states)

        if formula.operator == LTLOperator.UNTIL:
            return self.validate_until(
                str(formula.operands[0]),
                str(formula.operands[1]),
                states
            )

        if formula.operator == LTLOperator.NOT:
            valid, conf = self.validate_formula(
                LTLFormula(None, formula.operands), states
            )
            return not valid, 1 - conf

        if formula.operator == LTLOperator.AND:
            v1, c1 = self.validate_formula(
                formula.operands[0] if isinstance(formula.operands[0], LTLFormula)
                else LTLFormula(None, [formula.operands[0]]),
                states
            )
            v2, c2 = self.validate_formula(
                formula.operands[1] if isinstance(formula.operands[1], LTLFormula)
                else LTLFormula(None, [formula.operands[1]]),
                states
            )
            return v1 and v2, min(c1, c2)

        if formula.operator == LTLOperator.OR:
            v1, c1 = self.validate_formula(
                formula.operands[0] if isinstance(formula.operands[0], LTLFormula)
                else LTLFormula(None, [formula.operands[0]]),
                states
            )
            v2, c2 = self.validate_formula(
                formula.operands[1] if isinstance(formula.operands[1], LTLFormula)
                else LTLFormula(None, [formula.operands[1]]),
                states
            )
            return v1 or v2, max(c1, c2)

        return False, 0.0

    def from_bus_events(self, events: list[dict]) -> list[SuperpositionalState]:
        """Convert bus events to superpositional state sequence."""
        states = []

        for event in events:
            topic = event.get('topic', 'unknown')
            kind = event.get('kind', 'event')
            level = event.get('level', 'info')

            # Create superposition of possible interpretations
            basis = [
                f"topic:{topic}",
                f"kind:{kind}",
                f"level:{level}",
                f"active",
            ]

            # Weights based on event properties
            weights = [1.0, 0.8, 0.6, 1.0 if level != 'error' else 0.3]

            states.append(self.create_superposition(basis, weights))

        return states

# =============================================================================
# VEC2VEC TRANSFORMER
# =============================================================================

class Vec2VecTransformer:
    """Vector-to-vector object transformer."""

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        self.dim = dim
        self.transforms: dict[str, Vec2VecTransform] = {}
        self.rng = np.random.default_rng()

    def register_transform(
        self,
        name: str,
        source_space: str,
        target_space: str,
        matrix: np.ndarray | None = None,
        nonlinear: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> None:
        """Register a named transformation."""
        self.transforms[name] = Vec2VecTransform(
            source_space=source_space,
            target_space=target_space,
            transform_matrix=matrix,
            nonlinear_fn=nonlinear
        )

    def create_rotation_transform(self, name: str, angle: float) -> None:
        """Create rotation transformation (2D subspace)."""
        c, s = np.cos(angle), np.sin(angle)
        R = np.eye(self.dim)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c
        self.register_transform(name, "embedding", "rotated", matrix=R)

    def create_projection_transform(self, name: str, target_dim: int) -> None:
        """Create random projection to lower dimension."""
        proj = self.rng.standard_normal((target_dim, self.dim)) / np.sqrt(target_dim)
        self.register_transform(name, "embedding", f"projected_{target_dim}", matrix=proj)

    def create_attention_transform(self, name: str, query: np.ndarray) -> None:
        """Create attention-weighted transformation."""
        # Outer product creates attention pattern
        attention = np.outer(query, query)
        attention = attention / (np.linalg.norm(attention) + 1e-10)
        self.register_transform(
            name, "embedding", "attended",
            matrix=np.eye(self.dim) + 0.5 * attention
        )

    def apply(self, name: str, vec: np.ndarray) -> np.ndarray:
        """Apply named transformation."""
        if name not in self.transforms:
            raise ValueError(f"Unknown transform: {name}")
        return self.transforms[name].apply(vec)

    def compose(self, names: list[str], vec: np.ndarray) -> np.ndarray:
        """Apply sequence of transformations."""
        result = vec
        for name in names:
            result = self.apply(name, result)
        return result

    def cross_modal_transform(
        self,
        sextet: SextetVector,
        source_channel: SextetChannel,
        target_channel: SextetChannel
    ) -> np.ndarray:
        """Transform between sextet channels."""
        source = sextet.channel(source_channel)
        target = sextet.channel(target_channel)

        # Learn simple linear mapping
        # (In production, this would be learned from data)
        correlation = np.outer(target, source)
        transform = correlation / (np.linalg.norm(correlation) + 1e-10)

        return transform @ source

# =============================================================================
# MAIN ENGINE
# =============================================================================

class TokenGeometryEngine:
    """
    Unified token geometry engine combining:
    - BPE tokenization (tiktoken)
    - AUOM semantic analysis
    - Sextet multimodal encoding
    - N-sphere geometry
    - Superpositional LTL
    - Vec2Vec transformations
    """

    def __init__(
        self,
        use_tiktoken: bool = True,
        use_sbert: bool = False,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        nsphere_dim: int = NSPHERE_DIM,
        sextet_channel_dim: int = 64
    ):
        # Tokenizer
        if use_tiktoken and TIKTOKEN_AVAILABLE:
            self.tokenizer = TiktokenBackend()
        else:
            self.tokenizer = FallbackTokenizer()

        # Embedder
        if use_sbert and SBERT_AVAILABLE:
            self.embedder = SentenceTransformerBackend()
            embedding_dim = self.embedder.dim
        else:
            self.embedder = HashEmbedding(embedding_dim)

        # Components
        self.auom_analyzer = AUOMAnalyzer()
        self.sextet_encoder = SextetEncoder(sextet_channel_dim)
        self.nsphere = NSphereGeometry(nsphere_dim)
        self.ltl = SuperpositionalLTL()
        self.vec2vec = Vec2VecTransformer(embedding_dim)

        # Initialize standard transforms
        self._init_standard_transforms()

    def _init_standard_transforms(self):
        """Initialize standard vec2vec transformations."""
        self.vec2vec.create_rotation_transform("rotate_90", np.pi / 2)
        self.vec2vec.create_rotation_transform("rotate_45", np.pi / 4)
        self.vec2vec.create_projection_transform("project_64", 64)
        self.vec2vec.create_projection_transform("project_32", 32)

    def process(
        self,
        text: str,
        bus_events: list[dict] | None = None,
        ltl_formula: LTLFormula | None = None
    ) -> TokenGeometryResult:
        """
        Process text through full token geometry pipeline.

        Args:
            text: Input text to process
            bus_events: Optional bus events for LTL validation
            ltl_formula: Optional LTL formula to validate

        Returns:
            TokenGeometryResult with all computed representations
        """
        start_time = time.perf_counter()

        # 1. BPE Tokenization
        bpe_tokens = self.tokenizer.encode(text)
        token_count = len(bpe_tokens)

        # 2. AUOM Analysis
        auom_units = self.auom_analyzer.analyze(text, self.embedder)
        detected_operators = [
            u.metadata.get('operator')
            for u in auom_units
            if u.category == AUOMCategory.OPERATOR
        ]

        # 3. Sextet Encoding
        sextet_vector = self.sextet_encoder.encode(text, auom_units, self.embedder)

        # 4. N-Sphere Projection
        full_vector = sextet_vector.to_numpy()
        nsphere_point = self.nsphere.project_to_sphere(full_vector)

        # 5. Superpositional LTL Validation
        ltl_valid = True
        superposition = None

        if bus_events:
            states = self.ltl.from_bus_events(bus_events)
            if states:
                # Create superposition of current text state
                superposition = self.ltl.create_superposition(
                    [f"text:{text[:50]}", "processing", "active"],
                    [1.0, 0.8, 0.9]
                )

            if ltl_formula:
                ltl_valid, _ = self.ltl.validate_formula(ltl_formula, states)

        processing_time = (time.perf_counter() - start_time) * 1000

        return TokenGeometryResult(
            raw_text=text,
            bpe_tokens=bpe_tokens,
            token_count=token_count,
            auom_units=auom_units,
            detected_operators=detected_operators,
            sextet_vector=sextet_vector,
            nsphere_point=nsphere_point,
            ltl_formula=ltl_formula,
            ltl_valid=ltl_valid,
            superposition=superposition,
            processing_time_ms=processing_time
        )

    def count_tokens(self, text: str) -> int:
        """Quick token count without full processing."""
        return self.tokenizer.count(text)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        return self.tokenizer.truncate(text, max_tokens)

    def embed(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        return self.embedder.embed_single(text)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between texts."""
        v1 = self.embed(text1)
        v2 = self.embed(text2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

    def geodesic_similarity(self, text1: str, text2: str) -> float:
        """Compute geodesic similarity on n-sphere."""
        r1 = self.process(text1)
        r2 = self.process(text2)

        if r1.nsphere_point and r2.nsphere_point:
            dist = r1.nsphere_point.geodesic_distance(r2.nsphere_point)
            # Convert distance to similarity (0 = identical, π = opposite)
            return 1.0 - dist / np.pi
        return 0.0

    def to_bus_event(self, result: TokenGeometryResult) -> dict:
        """Convert result to bus event format."""
        return {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": "token_geometry.process",
            "kind": "metric",
            "level": "info",
            "actor": "token_geometry_engine",
            "data": {
                "token_count": result.token_count,
                "auom_count": len(result.auom_units),
                "detected_operators": result.detected_operators,
                "ltl_valid": result.ltl_valid,
                "processing_time_ms": result.processing_time_ms,
                "nsphere_coords": result.nsphere_point.coords[:8].tolist() if result.nsphere_point else None,
            }
        }

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Token Geometry Engine - Unified tokenization, embedding, and validation"
    )
    parser.add_argument("text", nargs="?", help="Text to process")
    parser.add_argument("--count", action="store_true", help="Only count tokens")
    parser.add_argument("--truncate", type=int, help="Truncate to N tokens")
    parser.add_argument("--similarity", nargs=2, help="Compute similarity between two texts")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--emit-bus", action="store_true", help="Emit result to bus")
    parser.add_argument("--bus-dir", default="/pluribus/.pluribus/bus", help="Bus directory")

    args = parser.parse_args()
    engine = TokenGeometryEngine()

    if args.similarity:
        sim = engine.similarity(args.similarity[0], args.similarity[1])
        geo_sim = engine.geodesic_similarity(args.similarity[0], args.similarity[1])
        print(f"Cosine similarity: {sim:.4f}")
        print(f"Geodesic similarity: {geo_sim:.4f}")
        return

    if not args.text:
        # Read from stdin
        args.text = sys.stdin.read().strip()

    if args.count:
        print(engine.count_tokens(args.text))
        return

    if args.truncate:
        print(engine.truncate_to_tokens(args.text, args.truncate))
        return

    result = engine.process(args.text)

    if args.json:
        output = {
            "token_count": result.token_count,
            "auom_units": [
                {"text": u.text, "category": u.category.value, "confidence": u.confidence}
                for u in result.auom_units
            ],
            "detected_operators": result.detected_operators,
            "ltl_valid": result.ltl_valid,
            "processing_time_ms": result.processing_time_ms,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Tokens: {result.token_count}")
        print(f"AUOM Units: {len(result.auom_units)}")
        for u in result.auom_units:
            print(f"  - [{u.category.value}] {u.text} ({u.confidence:.2f})")
        if result.detected_operators:
            print(f"Operators: {', '.join(result.detected_operators)}")
        print(f"LTL Valid: {result.ltl_valid}")
        print(f"Processing: {result.processing_time_ms:.2f}ms")

    if args.emit_bus:
        bus_path = Path(args.bus_dir) / "events.ndjson"
        bus_path.parent.mkdir(parents=True, exist_ok=True)
        event = engine.to_bus_event(result)
        with open(bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        print(f"Emitted to {bus_path}")

if __name__ == "__main__":
    main()
