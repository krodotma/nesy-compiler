#!/usr/bin/env python3
"""
LENS Entropy Profiler - Computes the 8-dimensional H* entropy vector.

This module implements the entropy profiling system for the LENS/LASER
Synthesizer pipeline as specified in lens_laser_synthesizer_v1.md.

The entropy vector H* captures multiple dimensions of response quality:
- Information density (h_info)
- Missing information (h_miss)
- Conjecture entropy (h_conj)
- Aleatoric uncertainty (h_alea)
- Epistemic uncertainty (h_epis)
- Structural entropy (h_struct)
- Cognitive load (c_load)
- Goal drift (h_goal_drift)

Author: claude-codex (Task 4.4 - Entelexis Architecture)
DKIN Version: v28
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.dont_write_bytecode = True


# ==============================================================================
# Constants and Configuration
# ==============================================================================

# Epistemic uncertainty markers (words indicating model uncertainty)
EPISTEMIC_MARKERS = frozenset([
    "unknown", "uncertain", "possibly", "perhaps", "maybe", "might",
    "could", "likely", "unlikely", "probably", "presumably", "apparently",
    "allegedly", "reportedly", "supposedly", "unclear", "ambiguous",
    "debatable", "questionable", "speculative", "hypothetically"
])

# Conjecture markers (words indicating unverifiable claims)
CONJECTURE_MARKERS = frozenset([
    "always", "never", "definitely", "certainly", "absolutely",
    "undoubtedly", "unquestionably", "invariably", "obviously",
    "clearly", "evidently", "surely", "doubtless", "indubitably",
    "all experts agree", "everyone knows", "it is well known"
])

# Abstract concept markers (increase cognitive load)
ABSTRACT_MARKERS = frozenset([
    "concept", "paradigm", "framework", "methodology", "ontology",
    "epistemology", "phenomenology", "hermeneutics", "dialectic",
    "abstraction", "instantiation", "reification", "hypostasis",
    "supervenience", "emergence", "entailment", "presupposition",
    "schema", "archetype", "prototype", "morphism", "functor"
])

# Default constraint budgets (from spec Section 6.2)
DEFAULT_BUDGETS = {
    "conjecture_max": 0.05,
    "missing_max": 0.10,
    "cogload_max": 0.30,
    "goal_drift_max": 0.10,
    "info_min": 0.50
}


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class EntropyVector:
    """
    8-dimensional entropy vector for response quality measurement.

    Components (all in [0, 1]):
    - h_info: Information density (grounded, prompt-relevant signal)
    - h_miss: Missing information (omitted required information)
    - h_conj: Conjecture entropy (unsupported assertions as fact)
    - h_alea: Aleatoric uncertainty (irreducible sampling stochasticity)
    - h_epis: Epistemic uncertainty (model knowledge gaps)
    - h_struct: Structural entropy (rhetorical overhead)
    - c_load: Cognitive load (human processing cost)
    - h_goal_drift: Goal drift (divergence from user intent)

    Computed fields:
    - h_total: Sum of all 8 components
    - h_mean: h_total / 8
    - utility: U(Y) = h_info * PROD(1 - H_i) / (1 + c_load)
    """
    # Primary entropy components
    h_info: float = 0.0
    h_miss: float = 0.0
    h_conj: float = 0.0
    h_alea: float = 0.0
    h_epis: float = 0.0
    h_struct: float = 0.0
    c_load: float = 0.0
    h_goal_drift: float = 0.0

    # Computed fields (set by __post_init__)
    h_total: float = field(init=False)
    h_mean: float = field(init=False)
    utility: float = field(init=False)

    def __post_init__(self):
        """Compute derived fields after initialization."""
        # Clamp all values to [0, 1]
        self.h_info = _clamp(self.h_info)
        self.h_miss = _clamp(self.h_miss)
        self.h_conj = _clamp(self.h_conj)
        self.h_alea = _clamp(self.h_alea)
        self.h_epis = _clamp(self.h_epis)
        self.h_struct = _clamp(self.h_struct)
        self.c_load = _clamp(self.c_load)
        self.h_goal_drift = _clamp(self.h_goal_drift)

        # Compute totals
        self.h_total = (
            self.h_info + self.h_miss + self.h_conj + self.h_alea +
            self.h_epis + self.h_struct + self.c_load + self.h_goal_drift
        )
        self.h_mean = self.h_total / 8.0

        # Compute utility using the quality functional
        # U(Y) = H_info * PROD(1 - H_i) / (1 + C_load)
        self.utility = compute_utility(self)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "h_info": round(self.h_info, 4),
            "h_miss": round(self.h_miss, 4),
            "h_conj": round(self.h_conj, 4),
            "h_alea": round(self.h_alea, 4),
            "h_epis": round(self.h_epis, 4),
            "h_struct": round(self.h_struct, 4),
            "c_load": round(self.c_load, 4),
            "h_goal_drift": round(self.h_goal_drift, 4),
            "h_total": round(self.h_total, 4),
            "h_mean": round(self.h_mean, 4),
            "utility": round(self.utility, 4)
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "EntropyVector":
        """Create from dictionary."""
        return cls(
            h_info=d.get("h_info", 0.0),
            h_miss=d.get("h_miss", 0.0),
            h_conj=d.get("h_conj", 0.0),
            h_alea=d.get("h_alea", 0.0),
            h_epis=d.get("h_epis", 0.0),
            h_struct=d.get("h_struct", 0.0),
            c_load=d.get("c_load", 0.0),
            h_goal_drift=d.get("h_goal_drift", 0.0)
        )

    def interpret_quality(self) -> str:
        """
        Interpret the H_mean value as quality tier.

        From spec Section 2.3:
        - H_mean = 0.0-0.1: Near-ideal (tight, grounded, minimal noise)
        - H_mean = 0.2-0.3: Expert-dense but cognitively heavy
        - H_mean = 0.4-0.5: Mediocre / noisy
        - H_mean > 0.6: Unreliable or bloated
        """
        if self.h_mean <= 0.1:
            return "near_ideal"
        elif self.h_mean <= 0.3:
            return "expert_dense"
        elif self.h_mean <= 0.5:
            return "mediocre"
        else:
            return "unreliable"

    def clarity(self) -> float:
        """Compute clarity score (1 - H_mean)."""
        return 1.0 - self.h_mean


# ==============================================================================
# Utility Functions
# ==============================================================================

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp value to [low, high] range."""
    return max(low, min(high, value))


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization with punctuation handling."""
    # Remove markdown code blocks for content analysis
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    # Tokenize
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def _sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences."""
    # Handle common abbreviations
    text = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', '\n', text)
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    return sentences if sentences else [text]


def _count_word_occurrences(text: str, word_set: frozenset[str]) -> int:
    """Count occurrences of words from a set in text."""
    tokens = _tokenize(text)
    return sum(1 for token in tokens if token in word_set)


def _compute_compression_ratio(text: str) -> float:
    """
    Compute compression ratio as proxy for information density.
    Higher compression = more redundancy = lower information density.
    """
    if not text:
        return 0.0
    original = text.encode('utf-8')
    compressed = zlib.compress(original, level=9)
    if len(original) == 0:
        return 0.0
    ratio = len(compressed) / len(original)
    return ratio


def _extract_claims(text: str) -> list[str]:
    """
    Extract atomic claims from text.
    Heuristic: sentences that contain assertions (not questions, not imperatives).
    """
    sentences = _sentence_tokenize(text)
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Skip questions
        if sent.endswith('?'):
            continue
        # Skip imperatives (starting with verb)
        if re.match(r'^(please|try|do|don\'t|let|make|run|use|add|remove)\b', sent.lower()):
            continue
        # Skip very short sentences
        if len(sent.split()) < 3:
            continue
        claims.append(sent)
    return claims


def _compute_jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _count_topic_transitions(text: str, window_size: int = 50) -> int:
    """
    Count topic transitions (concept switches) using sliding window.
    Heuristic: significant vocabulary change between windows.
    """
    tokens = _tokenize(text)
    if len(tokens) < window_size * 2:
        return 0

    transitions = 0
    prev_window_set: set[str] = set()

    for i in range(0, len(tokens) - window_size, window_size // 2):
        current_window = tokens[i:i + window_size]
        current_set = set(current_window)

        if prev_window_set:
            # Significant vocabulary change indicates topic transition
            similarity = _compute_jaccard_similarity(prev_window_set, current_set)
            if similarity < 0.3:  # Less than 30% overlap
                transitions += 1

        prev_window_set = current_set

    return transitions


# ==============================================================================
# Entropy Estimator Functions
# ==============================================================================

def estimate_h_info(Y: str, X: str) -> float:
    """
    Estimate information density (H_info).

    Measures the ratio of prompt-relevant tokens to total tokens.

    Args:
        Y: Response text
        X: Prompt text

    Returns:
        float: Information density in [0, 1]
    """
    if not Y or not X:
        return 0.0

    # Tokenize both prompt and response
    prompt_tokens = set(_tokenize(X))
    response_tokens = _tokenize(Y)

    if not response_tokens:
        return 0.0

    # Count how many response tokens relate to prompt content
    # This is a heuristic: exact match or contains prompt terms
    relevant_count = 0

    for token in response_tokens:
        # Token directly from prompt
        if token in prompt_tokens:
            relevant_count += 1
        # Token is a derivative (e.g., "entropy" -> "entropic")
        elif any(token.startswith(pt[:4]) for pt in prompt_tokens if len(pt) >= 4):
            relevant_count += 0.5

    # Normalize by response length
    base_ratio = relevant_count / len(response_tokens)

    # Boost for compression efficiency (high compression = low redundancy = high info)
    compression = _compute_compression_ratio(Y)
    # Good compression ratio is around 0.3-0.5
    compression_bonus = 1.0 - abs(compression - 0.4) if compression > 0 else 0.0

    # Combine signals
    h_info = base_ratio * 0.7 + compression_bonus * 0.3

    return _clamp(h_info)


def estimate_h_miss(Y: str, schema: dict[str, Any] | None) -> float:
    """
    Estimate missing information (H_miss).

    Measures what proportion of required schema slots are unfilled.

    Args:
        Y: Response text
        schema: Task schema with required_slots

    Returns:
        float: Missing information ratio in [0, 1]
    """
    if not schema:
        return 0.0

    required_slots = schema.get("required_slots", [])
    if not required_slots:
        return 0.0

    y_lower = Y.lower() if Y else ""
    y_tokens = set(_tokenize(Y)) if Y else set()
    filled = 0

    for slot in required_slots:
        slot_name = slot if isinstance(slot, str) else str(slot)
        slot_lower = slot_name.lower()

        # Generate slot variants for matching
        slot_variants = [
            slot_lower,
            slot_lower.replace("_", " "),
            slot_lower.replace("_", ""),
            slot_lower.replace("-", " "),
            slot_lower.replace("-", ""),
        ]

        # Check if slot name appears directly
        if any(variant in y_lower for variant in slot_variants):
            filled += 1
            continue

        # Check if slot tokens appear (more lenient)
        slot_tokens = set(_tokenize(slot_name))
        if slot_tokens and slot_tokens & y_tokens:
            # At least some slot tokens present
            overlap = len(slot_tokens & y_tokens) / len(slot_tokens)
            if overlap >= 0.5:
                filled += 1
                continue
            filled += overlap  # Partial credit

        # Check semantic similarity via keyword expansion
        slot_keywords = {
            "definition": {"define", "means", "is", "refers", "describes"},
            "formula": {"equation", "formula", "expression", "calculate", "=", "sum"},
            "example": {"example", "instance", "such as", "for example", "e.g"},
            "explanation": {"explain", "because", "reason", "why", "how"},
            "relationship": {"relation", "connect", "link", "between", "and"},
        }
        if slot_lower in slot_keywords:
            if slot_keywords[slot_lower] & y_tokens:
                filled += 0.7  # Semantic match gives partial credit

    missing_ratio = (len(required_slots) - filled) / len(required_slots)
    return _clamp(missing_ratio)


def estimate_h_conj(Y: str) -> float:
    """
    Estimate conjecture entropy (H_conj).

    Measures the ratio of unverifiable/overconfident assertions.
    Uses heuristics to detect claims presented without evidence.

    Args:
        Y: Response text

    Returns:
        float: Conjecture ratio in [0, 1]
    """
    if not Y:
        return 0.0

    claims = _extract_claims(Y)
    if not claims:
        return 0.0

    unverifiable_count = 0

    for claim in claims:
        claim_lower = claim.lower()

        # Check for overconfident markers (absolutist language without evidence)
        has_conjecture_marker = any(
            marker in claim_lower for marker in CONJECTURE_MARKERS
        )

        # Check for citation/evidence markers
        has_evidence = any(marker in claim_lower for marker in [
            "according to", "research shows", "studies indicate",
            "evidence suggests", "data shows", "experiment",
            "measured", "observed", "documented", "cited", "source"
        ])

        # Unverifiable if overconfident without evidence
        if has_conjecture_marker and not has_evidence:
            unverifiable_count += 1

        # Also flag very long sentences without qualifiers (potentially overconfident)
        word_count = len(claim.split())
        if word_count > 30 and not has_evidence:
            has_qualifier = any(
                q in claim_lower for q in ["may", "might", "could", "possibly", "likely"]
            )
            if not has_qualifier:
                unverifiable_count += 0.5

    return _clamp(unverifiable_count / len(claims))


def estimate_h_alea(samples: list[str] | None) -> float:
    """
    Estimate aleatoric uncertainty (H_alea).

    Measures variance across multiple samples of the same prompt.
    High variance = high aleatoric uncertainty.

    Args:
        samples: List of response samples from resampling

    Returns:
        float: Aleatoric uncertainty in [0, 1]
    """
    if not samples or len(samples) < 2:
        return 0.0

    # Convert samples to token sets for comparison
    sample_sets = [set(_tokenize(s)) for s in samples]

    # Compute pairwise Jaccard similarities
    similarities: list[float] = []
    for i in range(len(sample_sets)):
        for j in range(i + 1, len(sample_sets)):
            sim = _compute_jaccard_similarity(sample_sets[i], sample_sets[j])
            similarities.append(sim)

    if not similarities:
        return 0.0

    # Mean similarity - high similarity means low aleatoric uncertainty
    mean_similarity = sum(similarities) / len(similarities)

    # Compute variance of similarities
    variance = sum((s - mean_similarity) ** 2 for s in similarities) / len(similarities)

    # Higher variance and lower similarity = higher aleatoric uncertainty
    # Transform to [0, 1] scale
    h_alea = (1.0 - mean_similarity) * 0.7 + math.sqrt(variance) * 0.3

    return _clamp(h_alea)


def estimate_h_epis(model_responses: list[str] | None) -> float:
    """
    Estimate epistemic uncertainty (H_epis).

    Measures disagreement between different models on the same prompt.
    Uses both explicit uncertainty markers and cross-model agreement.

    Args:
        model_responses: List of responses from different models

    Returns:
        float: Epistemic uncertainty in [0, 1]
    """
    if not model_responses:
        return 0.0

    # If only one response, measure explicit epistemic markers
    if len(model_responses) == 1:
        Y = model_responses[0]
        tokens = _tokenize(Y)
        if not tokens:
            return 0.0
        marker_count = _count_word_occurrences(Y, EPISTEMIC_MARKERS)
        marker_ratio = marker_count / len(tokens)
        # Scale up since markers are relatively rare
        return _clamp(marker_ratio * 5.0)

    # Multiple models: measure disagreement
    response_sets = [set(_tokenize(r)) for r in model_responses]

    # Compute pairwise disagreement
    disagreements: list[float] = []
    for i in range(len(response_sets)):
        for j in range(i + 1, len(response_sets)):
            sim = _compute_jaccard_similarity(response_sets[i], response_sets[j])
            disagreements.append(1.0 - sim)

    if not disagreements:
        return 0.0

    # Mean disagreement
    mean_disagreement = sum(disagreements) / len(disagreements)

    # Also check for explicit epistemic markers in responses
    total_tokens = sum(len(_tokenize(r)) for r in model_responses)
    total_markers = sum(_count_word_occurrences(r, EPISTEMIC_MARKERS) for r in model_responses)
    marker_ratio = total_markers / total_tokens if total_tokens > 0 else 0.0

    # Combine disagreement and marker signals
    h_epis = mean_disagreement * 0.6 + marker_ratio * 4.0 * 0.4

    return _clamp(h_epis)


def estimate_h_struct(Y: str) -> float:
    """
    Estimate structural entropy (H_struct).

    Measures rhetorical overhead using pattern-based heuristics.
    High structural entropy = bloated, redundant response with excessive
    transitional phrases and rhetorical padding.

    Args:
        Y: Response text

    Returns:
        float: Structural entropy in [0, 1]
    """
    if not Y:
        return 0.0

    tokens = _tokenize(Y)
    if not tokens:
        return 0.0

    h_struct = 0.0

    # Count rhetorical overhead patterns
    overhead_patterns = [
        # Transitional fillers
        (r'\b(however|moreover|furthermore|additionally|consequently|thus|hence|therefore)\b', 0.015),
        # Redundant clarifications
        (r'\b(in other words|that is to say|to put it differently|simply put|basically)\b', 0.025),
        # Self-references
        (r'\b(as mentioned|as stated|as discussed|as noted|as we said)\b', 0.02),
        # Hedging chains
        (r'\b(it is worth noting that|it should be noted that|importantly)\b', 0.02),
        # Empty introductions
        (r'\b(the fact that|the thing is|what this means is)\b', 0.015),
        # Redundant emphasis
        (r'\b(actually|literally|really|truly|very|extremely)\b', 0.01),
    ]

    for pattern, weight in overhead_patterns:
        matches = len(re.findall(pattern, Y, re.IGNORECASE))
        h_struct += matches * weight

    # Normalize by token count (longer text naturally has more patterns)
    # But also penalize if density is high
    pattern_density = h_struct / (len(tokens) / 100 + 1)
    h_struct = min(h_struct, pattern_density * 2)

    # Check for repetition (same phrases appearing multiple times)
    # Use 3-gram analysis
    if len(tokens) >= 6:
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        unique_trigrams = set(trigrams)
        if trigrams:
            repetition_ratio = 1.0 - (len(unique_trigrams) / len(trigrams))
            h_struct += repetition_ratio * 0.3

    # Compression-based component for longer texts (more reliable with more data)
    if len(Y) > 200:
        compression_ratio = _compute_compression_ratio(Y)
        # High compression = redundant; expect 0.3-0.5 for good text
        if compression_ratio > 0.5:
            h_struct += (compression_ratio - 0.5) * 0.4
        elif compression_ratio < 0.25:
            # Unusually high information density, slightly penalize
            h_struct += 0.1

    # Good structure indicators (reduce h_struct)
    # Paragraph breaks indicate organization
    paragraph_count = len(re.findall(r'\n\n|\n\s*\n', Y))
    if paragraph_count > 0 and len(tokens) > 50:
        h_struct -= min(0.1, paragraph_count * 0.02)

    return _clamp(h_struct)


def estimate_c_load(Y: str) -> float:
    """
    Estimate cognitive load (C_load).

    Measures human processing cost based on:
    - Concept switches (topic transitions)
    - Abstract terminology
    - Sentence complexity

    Args:
        Y: Response text

    Returns:
        float: Cognitive load in [0, 1]
    """
    if not Y:
        return 0.0

    tokens = _tokenize(Y)
    if not tokens:
        return 0.0

    # Count concept switches
    concept_switches = _count_topic_transitions(Y)

    # Count abstract terms
    abstract_count = _count_word_occurrences(Y, ABSTRACT_MARKERS)

    # Measure sentence complexity (average words per sentence)
    sentences = _sentence_tokenize(Y)
    avg_sentence_length = len(tokens) / len(sentences) if sentences else 0

    # Normalize components
    switch_ratio = concept_switches / (len(tokens) / 50 + 1)
    abstract_ratio = abstract_count / len(tokens)
    complexity_ratio = (avg_sentence_length - 15) / 30 if avg_sentence_length > 15 else 0

    # Weighted combination
    c_load = (
        switch_ratio * 0.3 +
        abstract_ratio * 5.0 * 0.4 +  # Scale up since abstract terms are rare
        complexity_ratio * 0.3
    )

    return _clamp(c_load)


def estimate_h_goal_drift(Y: str, X: str) -> float:
    """
    Estimate goal drift (H_goal_drift).

    Measures divergence from user intent using token overlap heuristics.
    Ideally would use embeddings, but this provides a reasonable proxy.

    Args:
        Y: Response text
        X: Prompt text

    Returns:
        float: Goal drift in [0, 1]
    """
    if not Y or not X:
        return 0.5  # Uncertain without both texts

    # Extract key terms from prompt (nouns, verbs)
    prompt_tokens = set(_tokenize(X))
    response_tokens = set(_tokenize(Y))

    # Filter to significant words (remove common stopwords)
    stopwords = frozenset([
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "and", "but", "if", "or", "because", "until", "while",
        "this", "that", "these", "those", "it", "its", "i", "you", "he", "she",
        "we", "they", "what", "which", "who", "whom", "your", "my", "his", "her",
        "our", "their", "me", "him", "us", "them"
    ])

    prompt_significant = prompt_tokens - stopwords
    response_significant = response_tokens - stopwords

    if not prompt_significant:
        return 0.0  # Can't measure drift without prompt content

    # Measure coverage of prompt terms in response
    coverage = len(prompt_significant & response_significant) / len(prompt_significant)

    # Measure response focus (how much of response relates to prompt)
    focus = len(prompt_significant & response_significant) / len(response_significant) if response_significant else 0

    # Goal drift is inverse of coverage and focus
    h_goal_drift = 1.0 - (coverage * 0.6 + focus * 0.4)

    return _clamp(h_goal_drift)


def compute_utility(H: EntropyVector | None = None, **kwargs: float) -> float:
    """
    Compute utility score using the quality functional.

    Formula from spec Section 2.2:
    U(Y) = H_info * PROD_{i in {miss,conj,alea,epis,struct,goal-drift}}(1 - H_i) / (1 + C_load)

    Args:
        H: EntropyVector instance, or None to use kwargs
        **kwargs: Individual entropy components if H is None

    Returns:
        float: Utility score in [0, 1]
    """
    if H is not None:
        h_info = H.h_info
        h_miss = H.h_miss
        h_conj = H.h_conj
        h_alea = H.h_alea
        h_epis = H.h_epis
        h_struct = H.h_struct
        c_load = H.c_load
        h_goal_drift = H.h_goal_drift
    else:
        h_info = kwargs.get("h_info", 0.0)
        h_miss = kwargs.get("h_miss", 0.0)
        h_conj = kwargs.get("h_conj", 0.0)
        h_alea = kwargs.get("h_alea", 0.0)
        h_epis = kwargs.get("h_epis", 0.0)
        h_struct = kwargs.get("h_struct", 0.0)
        c_load = kwargs.get("c_load", 0.0)
        h_goal_drift = kwargs.get("h_goal_drift", 0.0)

    # Product of (1 - H_i) for failure modes
    # Any failure mode approaching 1 collapses the product to 0
    product = (
        (1.0 - h_miss) *
        (1.0 - h_conj) *
        (1.0 - h_alea) *
        (1.0 - h_epis) *
        (1.0 - h_struct) *
        (1.0 - h_goal_drift)
    )

    # Utility formula
    utility = h_info * product / (1.0 + c_load)

    return _clamp(utility)


# ==============================================================================
# Main Profiling Function
# ==============================================================================

def profile_entropy(
    Y: str,
    X: str,
    samples: list[str] | None = None,
    model_responses: list[str] | None = None,
    schema: dict[str, Any] | None = None,
    emit_event: bool = True,
    bus_dir: str | None = None,
    actor: str = "lens-entropy-profiler"
) -> EntropyVector:
    """
    Profile the entropy of a response.

    Computes the full 8-dimensional entropy vector for a response Y
    given prompt X and optional additional data.

    Args:
        Y: Response text to profile
        X: Original prompt text
        samples: Optional list of resampled responses (for H_alea)
        model_responses: Optional list of responses from different models (for H_epis)
        schema: Optional task schema with required_slots (for H_miss)
        emit_event: Whether to emit bus event
        bus_dir: Bus directory path
        actor: Actor name for bus event

    Returns:
        EntropyVector: Complete entropy profile
    """
    # Compute all entropy components
    h_info = estimate_h_info(Y, X)
    h_miss = estimate_h_miss(Y, schema)
    h_conj = estimate_h_conj(Y)
    h_alea = estimate_h_alea(samples)
    h_epis = estimate_h_epis(model_responses or ([Y] if Y else None))
    h_struct = estimate_h_struct(Y)
    c_load = estimate_c_load(Y)
    h_goal_drift = estimate_h_goal_drift(Y, X)

    # Create entropy vector (computes h_total, h_mean, utility)
    vector = EntropyVector(
        h_info=h_info,
        h_miss=h_miss,
        h_conj=h_conj,
        h_alea=h_alea,
        h_epis=h_epis,
        h_struct=h_struct,
        c_load=c_load,
        h_goal_drift=h_goal_drift
    )

    # Emit bus event if requested
    if emit_event:
        _emit_profile_event(vector, Y, X, bus_dir, actor)

    return vector


def _emit_profile_event(
    vector: EntropyVector,
    Y: str,
    X: str,
    bus_dir: str | None,
    actor: str
) -> None:
    """Emit entropy.profile.computed bus event."""
    if bus_dir is None:
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")

    bus_path = Path(bus_dir) / "events.ndjson"

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": "entropy.profile.computed",
        "kind": "metric",
        "level": "info",
        "actor": actor,
        "data": {
            "entropy_vector": vector.to_dict(),
            "quality_tier": vector.interpret_quality(),
            "clarity": round(vector.clarity(), 4),
            "prompt_length": len(X) if X else 0,
            "response_length": len(Y) if Y else 0
        }
    }

    try:
        bus_path.parent.mkdir(parents=True, exist_ok=True)
        with bus_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass  # Non-fatal: bus emission is best-effort


# ==============================================================================
# CLI Interface
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for CLI."""
    p = argparse.ArgumentParser(
        prog="lens_entropy_profiler.py",
        description="LENS Entropy Profiler - Compute 8-dimensional H* entropy vector."
    )
    p.add_argument(
        "--response", "-r",
        required=True,
        help="Response text to profile (or @file.txt to read from file)"
    )
    p.add_argument(
        "--prompt", "-p",
        required=True,
        help="Prompt text (or @file.txt to read from file)"
    )
    p.add_argument(
        "--samples",
        nargs="*",
        help="Sample responses for aleatoric uncertainty (space-separated or @file.txt)"
    )
    p.add_argument(
        "--model-responses",
        nargs="*",
        help="Model responses for epistemic uncertainty (space-separated or @file.txt)"
    )
    p.add_argument(
        "--schema",
        help="Task schema JSON (or @file.json)"
    )
    p.add_argument(
        "--bus-dir",
        default=None,
        help="Bus directory (default: $PLURIBUS_BUS_DIR or /pluribus/.pluribus/bus)"
    )
    p.add_argument(
        "--actor",
        default="lens-entropy-profiler",
        help="Actor name for bus events"
    )
    p.add_argument(
        "--no-emit",
        action="store_true",
        help="Do not emit bus events"
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    return p


def _load_text(value: str) -> str:
    """Load text from string or @file reference."""
    if value.startswith("@"):
        path = Path(value[1:])
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"File not found: {path}")
    return value


def _load_json(value: str | None) -> dict[str, Any] | None:
    """Load JSON from string or @file reference."""
    if not value:
        return None
    if value.startswith("@"):
        path = Path(value[1:])
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(value)


def main(argv: list[str]) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)

    try:
        response = _load_text(args.response)
        prompt = _load_text(args.prompt)
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    samples = None
    if args.samples:
        samples = [_load_text(s) for s in args.samples]

    model_responses = None
    if args.model_responses:
        model_responses = [_load_text(r) for r in args.model_responses]

    schema = None
    if args.schema:
        try:
            schema = _load_json(args.schema)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            sys.stderr.write(f"Error loading schema: {e}\n")
            return 1

    vector = profile_entropy(
        Y=response,
        X=prompt,
        samples=samples,
        model_responses=model_responses,
        schema=schema,
        emit_event=not args.no_emit,
        bus_dir=args.bus_dir,
        actor=args.actor
    )

    if args.json:
        output = {
            "entropy_vector": vector.to_dict(),
            "quality_tier": vector.interpret_quality(),
            "clarity": round(vector.clarity(), 4)
        }
        sys.stdout.write(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    else:
        print(f"LENS Entropy Profile")
        print(f"=" * 40)
        print(f"H_info (information):    {vector.h_info:.4f}")
        print(f"H_miss (missing):        {vector.h_miss:.4f}")
        print(f"H_conj (conjecture):     {vector.h_conj:.4f}")
        print(f"H_alea (aleatoric):      {vector.h_alea:.4f}")
        print(f"H_epis (epistemic):      {vector.h_epis:.4f}")
        print(f"H_struct (structural):   {vector.h_struct:.4f}")
        print(f"C_load (cognitive):      {vector.c_load:.4f}")
        print(f"H_goal_drift:            {vector.h_goal_drift:.4f}")
        print(f"-" * 40)
        print(f"H_total:                 {vector.h_total:.4f}")
        print(f"H_mean:                  {vector.h_mean:.4f}")
        print(f"Clarity (1 - H_mean):    {vector.clarity():.4f}")
        print(f"Utility:                 {vector.utility:.4f}")
        print(f"Quality tier:            {vector.interpret_quality()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
