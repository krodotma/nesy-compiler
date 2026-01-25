"""
Rhizome Eigenvalue Ranking (RER) Algorithm

A sophisticated evolutionary ranking algorithm for ARK/Pluribus that replaces
simple PageRank with a multi-dimensional, hysteresis-aware, semantically-rich
ranking system designed for negentropic evolution.

Key features:
- Temporal Hysteresis (DNA Axiom 4)
- Etymology propagation (semantic lineage)
- CMP fitness trajectories
- Thompson Sampling integration
- Spectral stability (Guard G6)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import logging

from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RERNodeState:
    """Extended state vector for a Rhizom node."""
    sha: str
    etymology_weight: float = 0.5      # η: Semantic origin strength
    cmp_fitness: float = 0.0           # φ: CMP score  
    entropy_norm: float = 1.0          # ||H*||: 8-dim entropy norm
    temporal_depth: int = 0            # τ: Generations from root
    omega_acceptance: float = 0.5      # ω: Büchi recurrence score
    beta_alpha: float = 1.0            # Thompson α
    beta_beta: float = 1.0             # Thompson β
    witness_count: int = 0             # Attestation count
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for matrix operations."""
        return np.array([
            self.etymology_weight,
            self.cmp_fitness,
            self.entropy_norm,
            self.temporal_depth / 100.0,  # Normalize depth
            self.omega_acceptance,
        ])


@dataclass
class RhizomeEigenvalueRanker:
    """
    RER: Rhizome Eigenvalue Ranking
    
    A DNA-aware evolutionary ranking algorithm that extends classic
    eigenvalue methods with:
    - Hysteresis memory (past states influence present)
    - Semantic etymology propagation
    - CMP fitness trajectories
    - Thompson Sampling for exploration
    - Spectral stability verification
    """
    
    damping: float = 0.85              # α: Damping factor
    hysteresis_weight: float = 0.15   # β: Past state influence
    hysteresis_depth: int = 10         # Memory depth
    max_iterations: int = 100
    tolerance: float = 1e-8
    
    # Internal state
    _historical_scores: List[np.ndarray] = field(default_factory=list)
    _transition_matrix: Optional[np.ndarray] = None
    _node_index: Dict[str, int] = field(default_factory=dict)
    
    def compute(self, dag: Any) -> Dict[str, float]:
        """
        Compute RER scores for all nodes in a RhizomDAG.
        
        Args:
            dag: RhizomDAG instance with nodes and edges
            
        Returns:
            Dict mapping SHA -> RER score (0-1, normalized)
        """
        logger.info("Computing RER scores for %d nodes", len(dag.nodes))
        
        # Build node index
        nodes = list(dag.nodes.values())
        self._node_index = {n['sha']: i for i, n in enumerate(nodes)}
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Build transition matrix
        self._transition_matrix = self._build_transition_matrix(dag, nodes)
        
        # Verify spectral stability
        if not self._check_spectral_stability():
            logger.warning("Transition matrix fails spectral stability - results may be unreliable")
        
        # Initialize scores (uniform or from CMP if available)
        initial_scores = np.array([
            max(0.1, n.get('cmp', 0.5)) for n in nodes
        ])
        initial_scores = initial_scores / np.sum(initial_scores)
        
        # Run power iteration with hysteresis
        final_scores = self._power_iteration(initial_scores)
        
        # Store for hysteresis
        self._historical_scores.append(final_scores.copy())
        if len(self._historical_scores) > self.hysteresis_depth:
            self._historical_scores.pop(0)
        
        # Build result dict
        return {
            nodes[i]['sha']: float(final_scores[i])
            for i in range(n)
        }
    
    def _build_transition_matrix(
        self, 
        dag: Any, 
        nodes: List[Dict]
    ) -> np.ndarray:
        """
        Build weighted semantic transition matrix.
        
        Edge weights incorporate:
        - Etymology similarity
        - CMP improvement
        - Entropy reduction
        - Witness attestations
        """
        n = len(nodes)
        T = np.zeros((n, n))
        
        for node in nodes:
            sha = node['sha']
            i = self._node_index[sha]
            
            # Get children
            children = dag.get_children(sha) if hasattr(dag, 'get_children') else []
            
            if not children:
                # Teleport from leaves (distribute to all)
                T[:, i] = 1.0 / n
            else:
                # Compute weighted edges to children
                weights = []
                indices = []
                
                for child_sha in children:
                    if child_sha not in self._node_index:
                        continue
                    j = self._node_index[child_sha]
                    child = dag.nodes.get(child_sha, {})
                    
                    weight = self._compute_edge_weight(node, child)
                    weights.append(weight)
                    indices.append(j)
                
                # Normalize outgoing weights
                total = sum(weights) if weights else 1.0
                for idx, w in zip(indices, weights):
                    T[idx, i] = w / total
        
        return T
    
    def _compute_edge_weight(
        self, 
        parent: Dict, 
        child: Dict
    ) -> float:
        """
        Compute transition weight based on DNA axioms.
        
        Factors:
        - Etymology propagation strength
        - CMP improvement (child > parent = good)
        - Entropy reduction (lower = better)
        - Witness attestation count
        """
        # Etymology similarity (simplified - could use embeddings)
        parent_etym = parent.get('etymology', '')
        child_etym = child.get('etymology', '')
        etymology_sim = self._etymology_similarity(parent_etym, child_etym)
        
        # CMP improvement factor
        parent_cmp = parent.get('cmp', 0.5)
        child_cmp = child.get('cmp', 0.5)
        cmp_delta = max(0, child_cmp - parent_cmp) / max(parent_cmp, 1e-6)
        
        # Entropy reduction factor
        parent_entropy = parent.get('entropy', {})
        child_entropy = child.get('entropy', {})
        parent_h = sum(parent_entropy.values()) if parent_entropy else 1.0
        child_h = sum(child_entropy.values()) if child_entropy else 1.0
        entropy_reduction = max(0, parent_h - child_h)
        
        # Witness boost
        witness_count = len(child.get('witnesses', []))
        witness_boost = 1.0 + 0.1 * witness_count
        
        return etymology_sim * (1 + cmp_delta) * (1 + entropy_reduction) * witness_boost
    
    def _etymology_similarity(self, a: str, b: str) -> float:
        """
        Simple etymology similarity based on word overlap.
        
        Could be enhanced with:
        - Embedding similarity
        - Semantic graph distance
        - AST-based similarity
        """
        if not a or not b:
            return 0.5  # Default for missing etymology
        
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        
        if not words_a or not words_b:
            return 0.5
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.5
    
    def _power_iteration(self, initial_scores: np.ndarray) -> np.ndarray:
        """
        Power iteration with hysteresis memory.
        
        r(t+1) = α * T * r(t) + (1-α) * teleport + β * hysteresis
        """
        n = len(initial_scores)
        r = initial_scores.copy()
        teleport = np.ones(n) / n
        
        for iteration in range(self.max_iterations):
            r_prev = r.copy()
            
            # Standard transition
            r = self.damping * self._transition_matrix @ r + (1 - self.damping) * teleport
            
            # Add hysteresis (decaying influence of past states)
            if self._historical_scores:
                hysteresis_term = np.zeros(n)
                total_weight = 0.0
                
                for i, hist in enumerate(reversed(self._historical_scores[-self.hysteresis_depth:])):
                    if len(hist) != n:
                        continue
                    decay = 0.9 ** (i + 1)
                    hysteresis_term += decay * hist
                    total_weight += decay
                
                if total_weight > 0:
                    hysteresis_term /= total_weight
                    r = (1 - self.hysteresis_weight) * r + self.hysteresis_weight * hysteresis_term
            
            # Normalize
            r_sum = np.sum(r)
            if r_sum > 0:
                r = r / r_sum
            
            # Check convergence
            if np.linalg.norm(r - r_prev) < self.tolerance:
                logger.debug("RER converged after %d iterations", iteration + 1)
                break
        
        return r
    
    def _check_spectral_stability(self) -> bool:
        """
        Verify transition matrix spectral stability (DNA Guard G6).
        
        Checks:
        - Dominant eigenvalue ≈ 1
        - Spectral gap for mixing
        - No problematic complex eigenvalues
        """
        if self._transition_matrix is None:
            return True
        
        try:
            eigenvalues = np.linalg.eigvals(self._transition_matrix)
            eigenvalues = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
            
            # Check dominant eigenvalue ≈ 1
            if abs(abs(eigenvalues[0]) - 1.0) > 1e-4:
                logger.warning("Dominant eigenvalue %.4f != 1", abs(eigenvalues[0]))
                return False
            
            # Check spectral gap
            if len(eigenvalues) > 1:
                spectral_gap = 1.0 - abs(eigenvalues[1])
                if spectral_gap < 0.05:
                    logger.warning("Small spectral gap: %.4f - slow convergence", spectral_gap)
            
            return True
            
        except Exception as e:
            logger.error("Spectral analysis failed: %s", e)
            return False
    
    def top_k(
        self, 
        scores: Dict[str, float], 
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top k nodes by RER score."""
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:k]
    
    def select_clade_weighted(
        self,
        clades: List[Any],
        scores: Dict[str, float],
        temperature: float = 1.0
    ) -> Any:
        """
        Select clade using RER-weighted Thompson Sampling.
        
        1. Sample from each clade's Beta posterior
        2. Weight by aggregate RER score
        3. Apply exploration temperature
        4. Select maximum
        
        Args:
            clades: List of Clade objects
            scores: RER scores dict (SHA -> score)
            temperature: Exploration temperature (>1 = more exploration)
        """
        if not clades:
            return None
        
        weighted_samples = []
        
        for clade in clades:
            # Thompson sample from Beta distribution
            alpha = getattr(clade, 'alpha', 1.0)
            beta = getattr(clade, 'beta', 1.0)
            sample = np.random.beta(alpha, beta)
            
            # Get aggregate RER score for clade members
            members = getattr(clade, 'members', [])
            if members:
                member_scores = [scores.get(m if isinstance(m, str) else getattr(m, 'sha', ''), 0.5) for m in members]
                clade_rer = np.mean(member_scores)
            else:
                clade_rer = 0.5
            
            # Apply temperature for exploration
            # Higher temperature = more uniform distribution
            weighted = (sample * (0.5 + 0.5 * clade_rer)) ** (1.0 / temperature)
            weighted_samples.append((clade, weighted))
        
        return max(weighted_samples, key=lambda x: x[1])[0]
    
    def select_from_archive(
        self,
        archive: Any,
        rer_scores: Dict[str, float],
        temperature: float = 1.0,
        diversity_weight: float = 0.3
    ) -> Any:
        """
        DGM-style parent selection from CladeArchive.
        
        Combines:
        - RER scores for ranked fitness
        - Lineage CMP for descendant productivity
        - Diversity bonus (inverse child count)
        - Thompson Sampling for exploration
        
        Args:
            archive: CladeArchive instance
            rer_scores: Pre-computed RER scores
            temperature: Exploration temperature
            diversity_weight: Weight for diversity bonus
            
        Returns:
            Selected Clade
        """
        if not hasattr(archive, 'nodes') or not archive.nodes:
            return None
        
        # Ensure lineage CMP is computed
        if hasattr(archive, 'recompute_all_lineage_cmp'):
            archive.recompute_all_lineage_cmp()
        
        clades = list(archive.nodes.values())
        weights = []
        
        for clade in clades:
            clade_id = clade.name
            
            # Thompson sample
            sample = np.random.beta(clade.alpha, clade.beta)
            
            # RER component (aggregate of members)
            members = getattr(clade, 'members', [])
            if members:
                member_rer = np.mean([rer_scores.get(m, 0.5) for m in members])
            else:
                member_rer = 0.5
            
            # Lineage CMP component
            lineage_cmp = archive._lineage_cmp_cache.get(clade_id, clade.cmp) if hasattr(archive, '_lineage_cmp_cache') else clade.cmp
            
            # Diversity component (inverse child count - DGM mechanism)
            child_count = getattr(clade, 'child_count', 0)
            if hasattr(archive, 'child_counts'):
                child_count = archive.child_counts.get(clade_id, 0)
            diversity_bonus = 1.0 / (1.0 + child_count)
            
            # Combined weight with temperature
            base_weight = (
                (1 - diversity_weight) * (sample * member_rer * lineage_cmp) +
                diversity_weight * diversity_bonus
            )
            
            # Apply temperature
            weight = base_weight ** (1.0 / temperature)
            weights.append(max(weight, 1e-10))
        
        # Softmax selection
        total = sum(weights)
        probs = [w / total for w in weights]
        
        selected = np.random.choice(clades, p=probs)
        
        logger.debug(
            "Selected clade %s from archive (temp=%.2f, diversity=%.2f)",
            selected.name, temperature, diversity_weight
        )
        
        return selected
    
    def get_inertia_score(self, sha: str, scores: Dict[str, float]) -> float:
        """
        Compute inertia score for a node (DNA Axiom 2).
        
        High RER = High inertia = Resistance to change
        
        This identifies "thermal mass" nodes that should not be
        mutated lightly (cf. GLM-4.7 lateral analysis).
        """
        base_rer = scores.get(sha, 0.5)
        
        # Add hysteresis component - nodes with stable historical scores
        # have higher inertia
        stability = 1.0
        if len(self._historical_scores) >= 2:
            idx = self._node_index.get(sha)
            if idx is not None:
                recent_scores = [h[idx] for h in self._historical_scores[-5:] if len(h) > idx]
                if len(recent_scores) >= 2:
                    stability = 1.0 - np.std(recent_scores)
        
        return base_rer * stability
    
    def propose_self_modification(
        self,
        clade: Any,
        rer_scores: Dict[str, float],
        archive: Any = None
    ) -> Dict[str, Any]:
        """
        DGM-style self-modification proposal.
        
        Analyzes clade state and proposes modifications based on:
        - Low-RER members (candidates for improvement)
        - Trait gaps vs successful ancestors
        - Stepping stone potential
        
        Returns proposal dict for MetaLearner integration.
        
        Args:
            clade: Clade proposing self-modification
            rer_scores: Current RER scores
            archive: Optional CladeArchive for ancestry analysis
            
        Returns:
            Dict with modification proposal metadata
        """
        members = getattr(clade, 'members', [])
        
        # Find weakest members by RER
        member_rer = [(m, rer_scores.get(m, 0.5)) for m in members]
        member_rer.sort(key=lambda x: x[1])
        
        weakest = member_rer[:3] if len(member_rer) >= 3 else member_rer
        
        # Analyze ancestry traits if archive available
        ancestor_traits = {}
        if archive and hasattr(clade, 'parent') and clade.parent:
            ancestors = archive.get_ancestors(clade.name) if hasattr(archive, 'get_ancestors') else []
            for anc_id in ancestors[:3]:  # Top 3 ancestors
                anc = archive.nodes.get(anc_id)
                if anc and hasattr(anc, 'traits'):
                    for trait, value in anc.traits.items():
                        if trait not in ancestor_traits:
                            ancestor_traits[trait] = []
                        ancestor_traits[trait].append(value)
        
        # Compute trait gaps
        trait_gaps = {}
        if ancestor_traits:
            clade_traits = getattr(clade, 'traits', {})
            for trait, anc_values in ancestor_traits.items():
                anc_avg = np.mean(anc_values)
                clade_val = clade_traits.get(trait, 0.5)
                if anc_avg > clade_val + 0.1:  # Ancestor was better
                    trait_gaps[trait] = anc_avg - clade_val
        
        # Check stepping stone potential
        is_stepping_stone = False
        if archive and hasattr(archive, 'stepping_stones'):
            is_stepping_stone = clade.name in archive.stepping_stones
        
        proposal = {
            "clade_id": clade.name,
            "clade_cmp": clade.cmp,
            "weakest_members": weakest,
            "trait_gaps": trait_gaps,
            "is_stepping_stone": is_stepping_stone,
            "suggested_focus": "trait_improvement" if trait_gaps else "member_optimization",
            "priority": "high" if len(weakest) > 0 and weakest[0][1] < 0.3 else "medium",
        }
        
        logger.info(
            "Self-modification proposal for %s: focus=%s, priority=%s",
            clade.name, proposal["suggested_focus"], proposal["priority"]
        )
        
        return proposal


# Convenience function
def compute_rer_scores(dag: Any, **kwargs) -> Dict[str, float]:
    """Compute RER scores for a RhizomDAG."""
    ranker = RhizomeEigenvalueRanker(**kwargs)
    return ranker.compute(dag)


def select_parent_from_archive(
    archive: Any,
    dag: Any = None,
    temperature: float = 1.0,
    **ranker_kwargs
) -> Any:
    """
    Convenience function: Select parent from CladeArchive using RER.
    
    Args:
        archive: CladeArchive instance
        dag: Optional RhizomDAG for RER computation
        temperature: Exploration temperature
        **ranker_kwargs: Additional args for RhizomeEigenvalueRanker
        
    Returns:
        Selected Clade
    """
    ranker = RhizomeEigenvalueRanker(**ranker_kwargs)
    
    # Compute RER scores if DAG provided
    rer_scores = {}
    if dag:
        rer_scores = ranker.compute(dag)
    
    return ranker.select_from_archive(archive, rer_scores, temperature)

