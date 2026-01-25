# Schmidhuber/DGM/HGM Gap Analysis for ARK

**Research Date**: 2026-01-23
**Swarm**: Claude (Constitutional), GLM-4.7 (Analysis)

---

## 1. Sources Analyzed

| Source | Key Contribution |
|--------|------------------|
| **Gödel Machine** (Schmidhuber 2006) | Self-improving AI via formal proof of self-modification benefit |
| **Darwin Gödel Machine (DGM)** (Sakana AI 2024) | Empirical validation over proofs, open-ended archive |
| **Huxley-Gödel Machine (HGM)** | CMP metric for descendant productivity, lineage trees |
| **CMP++ Proposal** | Hyperspherical geometry, spectral gating, HGT |
| **SOAR Framework** | LLM + evolutionary loop with hindsight learning |

---

## 2. ARK Current Implementation Status

### ✅ What ARK HAS (from `clade.py`):

| Mechanism | Implementation | Status |
|-----------|----------------|--------|
| Thompson Sampling | `sample_fitness()` using Beta(α, β) | ✅ Complete |
| CMP Fitness | `cmp` field on Clade | ✅ Basic |
| Trait Inheritance | `inherit_traits()` with variation | ✅ Complete |
| Success/Failure Recording | `record_success()` / `record_failure()` | ✅ Complete |
| Parent Reference | `parent: Optional[str]` | ✅ Basic |

### ❌ What ARK is MISSING:

| Mechanism | DGM/HGM Concept | Priority | Gap |
|-----------|-----------------|----------|-----|
| **Archive Tree** | Gene pool as growing tree of all past agents | 🔴 Critical | No archive structure |
| **Diversity Selection** | Blend of random + performance weighting | 🔴 Critical | Pure Thompson only |
| **Child Count Tracking** | Limit over-evolution from single node | 🟠 High | Not tracked |
| **Stepping Stones** | Keep low-scorers that enable future progress | 🔴 Critical | Only CMP ranking |
| **Descendant CMP** | CMP aggregates from ALL descendants, not just self | 🟠 High | Self-CMP only |
| **Self-Modification** | Agents rewrite their own code | 🟠 High | External mutation only |
| **Empirical Validation** | Test modifications on benchmarks | 🟢 Medium | LTL verification exists |
| **Lineage Visualization** | Tree view of evolution | 🟢 Low | No UI yet |

---

## 3. Key DGM Mechanisms Missing from ARK

### 3.1 Archive as Gene Pool (CRITICAL)

**DGM Behavior**:
- Maintains tree of ALL past agent versions
- Never discards failed agents ("stepping stones")
- Enables parallel exploration of design space
- Prevents premature convergence

**ARK Gap**:
- Rhizom DAG tracks commits, but no "agent archive"
- Clades can be pruned (dead-end removal)
- Missing: persistent archive that preserves ALL lineages

**Proposed Fix**:
```python
@dataclass
class CladeArchive:
    """DGM-style gene pool for agent lineages."""
    
    nodes: Dict[str, Clade] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (parent, child)
    
    def add_clade(self, clade: Clade, parent_id: Optional[str] = None) -> None:
        """Always add, never prune. Stepping stones preserved."""
        self.nodes[clade.name] = clade
        if parent_id:
            self.edges.append((parent_id, clade.name))
    
    def select_parent(self, temperature: float = 1.0) -> Clade:
        """
        DGM-style selection: blend of performance + diversity.
        - Higher CMP = higher selection probability
        - BUT low-child-count clades get bonus (diversity pressure)
        """
        child_counts = self._compute_child_counts()
        
        weights = []
        for name, clade in self.nodes.items():
            # Base weight from CMP
            w = clade.cmp ** temperature
            # Diversity bonus: fewer children = higher weight
            diversity_bonus = 1.0 / (1.0 + child_counts.get(name, 0))
            weights.append(w * (1 + diversity_bonus))
        
        # Softmax selection
        total = sum(weights)
        probs = [w / total for w in weights]
        
        selected = random.choices(list(self.nodes.values()), weights=probs, k=1)[0]
        return selected
```

### 3.2 Descendant CMP Aggregation (HIGH)

**DGM/HGM Behavior**:
- CMP is not just self-performance
- Aggregates benchmark outcomes from ALL descendants
- "Productive lineages" valued over "single-shot winners"

**ARK Gap**:
- `clade.cmp` is a single self-contained value
- No descendant aggregation mechanism
- Missing: lineage-level fitness function

**Proposed Fix**:
```python
def compute_lineage_cmp(
    archive: CladeArchive, 
    root_id: str,
    discount: float = 0.9
) -> float:
    """
    HGM-style CMP: aggregate descendants with temporal discount.
    
    CMP_lineage(n) = cmp(n) + γ * Σ CMP_lineage(children)
    """
    clade = archive.nodes[root_id]
    base_cmp = clade.cmp
    
    children = archive.get_children(root_id)
    if not children:
        return base_cmp
    
    descendant_cmp = sum(
        compute_lineage_cmp(archive, child, discount) 
        for child in children
    )
    
    return base_cmp + discount * descendant_cmp / len(children)
```

### 3.3 Stepping Stone Preservation (CRITICAL)

**DGM Behavior**:
- Low-scoring agents NOT discarded
- They may contain "stepping stones" - features that enable future breakthroughs
- Archive ablation showed removing this cripples performance

**ARK Gap**:
- Rhizom compaction plans to "remove low-CMP dead-ends"
- This is ANTI-DGM behavior!

**Proposed Fix**:
- Never delete clades from archive (soft-delete at most)
- Add `is_stepping_stone: bool` flag detected when:
  - Low self-CMP but high descendant-CMP
  - Contains novel traits not in ancestors
- Protect stepping stones from any pruning

### 3.4 Self-Modification Hooks (HIGH)

**DGM Behavior**:
- Agents can rewrite their OWN code via foundation model
- Example improvements: patch validation, multi-solution ranking, failure history

**ARK Gap**:
- Mutations come from external `DistillationPipeline`
- No mechanism for clade to propose self-modifications

**Proposed Fix**:
```python
class SelfModifyingClade(Clade):
    """DGM-style clade that can propose modifications to its own genes."""
    
    async def propose_modification(
        self, 
        meta_learner_client: MetaLearnerClient
    ) -> Optional["Gene"]:
        """
        Ask MetaLearner to suggest improvement to our weakest gene.
        """
        weakest_gene = min(self.genes, key=lambda g: g.fitness)
        
        suggestion = await meta_learner_client.suggest(
            context={
                "clade_traits": self.traits,
                "gene_to_improve": weakest_gene.to_dict(),
                "clade_history": self.get_history()
            },
            task="improve_gene"
        )
        
        if suggestion:
            return Gene.from_suggestion(suggestion)
        return None
```

---

## 4. Enhancement Priority Matrix

| Enhancement | Alignment | Effort | Impact | Priority |
|-------------|-----------|--------|--------|----------|
| Archive Tree Structure | DGM | Medium | High | **P0** |
| Descendant CMP Aggregation | HGM | Low | High | **P0** |
| Stepping Stone Detection | DGM | Low | Critical | **P0** |
| Diversity-Weighted Selection | DGM | Low | High | **P1** |
| Child Count Tracking | DGM | Trivial | Medium | **P1** |
| Self-Modification Hooks | DGM | High | High | **P2** |
| Lineage Visualization | DGM | Medium | Low | **P3** |

---

## 5. Recommended Implementation Order

### Phase A: Archive Foundation (P0)

1. **Create `CladeArchive` class** in `ribosome/archive.py`
2. **Add `child_count` field** to existing Clade
3. **Implement `compute_lineage_cmp()`** function
4. **Add stepping stone detection** logic
5. **REMOVE rhizome compaction** of clades (keep commits, but never delete clades)

### Phase B: Selection Enhancement (P1)

6. **Replace pure Thompson Sampling** with DGM-style blend:
   - 70% performance-weighted
   - 30% diversity bonus (inverse child count)
7. **Add exploration temperature** parameter
8. **Wire to `select_clade_weighted()` in RER**

### Phase C: Self-Modification (P2)

9. **Create `SelfModifyingClade` subclass**
10. **Wire to MetaLearner** for modification proposals
11. **Add empirical validation** via test suite execution
12. **Track modification history** per clade

---

## 6. Theoretical Alignment Check

| Schmidhuber Principle | ARK Mapping | Full Alignment? |
|-----------------------|-------------|-----------------|
| Formal proof of benefit | LTL verification | Partial - LTL is symbolic, not proof calculus |
| Self-referential improvement | Inception Controller | ✅ Yes |
| Utility maximization | CMP + RER | ✅ Yes |
| Infinite-horizon | Büchi acceptance, Hysteresis | ✅ Yes |
| Open-ended exploration | **MISSING** (need Archive) | ❌ No |
| Empirical validation | Test suite integration | ✅ Yes |

---

## 7. Conclusion

ARK has solid foundations from HGM (Thompson Sampling, CMP), but is **missing critical DGM mechanisms**:

1. **Archive-as-gene-pool** - Most critical gap
2. **Descendant CMP aggregation** - High impact, low effort
3. **Stepping stone preservation** - Counter to current compaction plans
4. **Diversity-weighted selection** - Simple enhancement to Thompson

**Recommendation**: Prioritize Phase A (Archive Foundation) before proceeding with Phase 2 MetaLearner integration. The archive is the backbone of open-ended evolution.

---

*Gap analysis by: Claude (Constitutional) + GLM-4.7 (Analysis)*
*Protocol: PBTSO-ARK-SCHMIDHUBER | Date: 2026-01-23*
