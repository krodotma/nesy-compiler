# Rhizome Eigenvalue Ranking (RER) Algorithm

## Overview

**RER** (Rhizome Eigenvalue Ranking) is a sophisticated evolutionary ranking algorithm 
designed specifically for ARK/Pluribus. Unlike PageRank which only considers link structure,
RER incorporates:

1. **Temporal Dynamics** - Past states influence present (Hysteresis Axiom)
2. **Semantic Etymology** - Meaning and purpose propagate through lineage
3. **Entropy Gradients** - 8-dimensional H* entropy vectors 
4. **CMP Trajectories** - Cumulative Meta-Priority fitness over time
5. **Thompson Sampling** - Bayesian clade selection with uncertainty

## Mathematical Foundation

### 1. State Vector per Node

Each Rhizom node `n` has an extended state vector:

```
S(n) = [
    η(n),        # Etymological weight (semantic origin strength)
    φ(n),        # CMP fitness score
    H*(n),       # 8-dim entropy vector norm
    τ(n),        # Temporal depth (generations from root)
    ω(n),        # Omega acceptance score (Büchi infinite recurrence)
    β_α(n), β_β(n)  # Thompson Sampling Beta parameters
]
```

### 2. Transition Matrix Construction

Unlike PageRank's simple adjacency, RER uses a **weighted semantic transition matrix**:

```python
def compute_transition_weight(parent: RhizomNode, child: RhizomNode) -> float:
    """
    Compute edge weight based on:
    - Etymology propagation strength
    - CMP improvement (positive = child improved on parent)
    - Entropy reduction factor
    - Witness attestation count
    """
    etymology_sim = semantic_similarity(parent.etymology, child.etymology)
    cmp_delta = max(0, child.cmp - parent.cmp) / max(parent.cmp, 1e-6)
    entropy_reduction = max(0, np.linalg.norm(parent.entropy) - np.linalg.norm(child.entropy))
    witness_boost = 1.0 + 0.1 * len(child.witnesses)
    
    return etymology_sim * (1 + cmp_delta) * (1 + entropy_reduction) * witness_boost
```

### 3. Power Iteration with Hysteresis

Standard power iteration, but incorporating past state influence:

```python
def rer_power_iteration(
    transition_matrix: np.ndarray,
    initial_scores: np.ndarray,
    historical_scores: List[np.ndarray],  # Hysteresis memory
    alpha: float = 0.85,                   # Damping factor
    beta: float = 0.15,                    # Hysteresis weight
    max_iterations: int = 100,
    tolerance: float = 1e-8
) -> np.ndarray:
    """
    RER ranking with hysteresis memory.
    
    r(t+1) = α * T * r(t) + (1-α) * teleport
           + β * Σ_i λ^i * r(t-i)  # Hysteresis term
    """
    n = len(initial_scores)
    r = initial_scores.copy()
    teleport = np.ones(n) / n
    
    for iteration in range(max_iterations):
        r_prev = r.copy()
        
        # Standard transition
        r = alpha * transition_matrix @ r + (1 - alpha) * teleport
        
        # Add hysteresis (decaying influence of past states)
        hysteresis_term = np.zeros(n)
        for i, hist in enumerate(reversed(historical_scores[-10:])):  # Last 10 states
            decay = 0.9 ** (i + 1)  # Exponential decay
            hysteresis_term += decay * hist
        
        if len(historical_scores) > 0:
            hysteresis_term /= sum(0.9 ** (i+1) for i in range(min(10, len(historical_scores))))
            r = (1 - beta) * r + beta * hysteresis_term
        
        # Normalize
        r = r / np.sum(r)
        
        # Check convergence
        if np.linalg.norm(r - r_prev) < tolerance:
            break
    
    return r
```

### 4. Thompson Sampling Integration

For clade selection, RER scores are combined with Thompson sampling:

```python
def select_clade_weighted(
    clades: List[Clade],
    rer_scores: Dict[str, float]
) -> Clade:
    """
    Select clade using RER-weighted Thompson Sampling.
    
    1. Sample from each clade's Beta posterior
    2. Weight by RER score
    3. Select max
    """
    weighted_samples = []
    
    for clade in clades:
        # Thompson sample
        sample = np.random.beta(clade.alpha, clade.beta)
        
        # Get aggregate RER score for clade
        clade_rer = np.mean([rer_scores.get(m.sha, 0.5) for m in clade.members])
        
        # Weighted score
        weighted = sample * (0.5 + 0.5 * clade_rer)
        weighted_samples.append((clade, weighted))
    
    return max(weighted_samples, key=lambda x: x[1])[0]
```

### 5. Spectral Stability Check

Before finalizing rankings, verify spectral stability (DNA Axiom G6):

```python
def check_spectral_stability(transition_matrix: np.ndarray) -> bool:
    """
    Verify the transition matrix satisfies spectral stability:
    - Dominant eigenvalue should be 1 (stochastic matrix property)
    - Second eigenvalue gap indicates mixing time
    - No unexpected complex eigenvalues with large magnitude
    """
    eigenvalues = np.linalg.eigvals(transition_matrix)
    eigenvalues = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
    
    # Check dominant eigenvalue ≈ 1
    if abs(abs(eigenvalues[0]) - 1.0) > 1e-6:
        return False
    
    # Check spectral gap (second eigenvalue should be < 0.95 for good mixing)
    spectral_gap = 1.0 - abs(eigenvalues[1])
    if spectral_gap < 0.05:
        logger.warning(f"Small spectral gap: {spectral_gap:.4f} - slow convergence")
    
    return True
```

## Key Advantages Over PageRank

| Aspect | PageRank | RER |
|--------|----------|-----|
| **Structure** | Link-only | Multi-dimensional state |
| **Time** | Memoryless | Hysteresis memory |
| **Semantics** | None | Etymology propagation |
| **Fitness** | None | CMP trajectory |
| **Uncertainty** | Deterministic | Thompson Sampling |
| **Evolution** | Static | Lineage-aware |

## Implementation Location

`nucleus/ark/rhizom/ranking.py` - Full RER implementation
`nucleus/ark/ribosome/clade.py` - Integration with clade selection

## DNA Axiom Alignment

| Axiom | RER Integration |
|-------|-----------------|
| **Entelecheia** | Etymology propagation measures purpose flow |
| **Inertia** | High RER nodes resist change (thermal mass) |
| **Hysteresis** | Historical score memory influences present |
| **Witness** | Witness attestations boost edge weights |
| **Infinity** | Omega acceptance scores for infinite traces |

## Usage Example

```python
from nucleus.ark.rhizom.ranking import RhizomeEigenvalueRanker

ranker = RhizomeEigenvalueRanker(
    damping=0.85,
    hysteresis_weight=0.15,
    hysteresis_depth=10
)

# Load Rhizom DAG
dag = RhizomDAG.load(repo_path)

# Compute RER scores
scores = ranker.compute(dag)

# Get top nodes by evolutionary fitness
top_nodes = ranker.top_k(scores, k=10)

# Use for clade selection
selected_clade = ranker.select_clade_weighted(clades, scores)
```
