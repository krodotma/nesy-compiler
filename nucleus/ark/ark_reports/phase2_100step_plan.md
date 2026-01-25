# ARK Phase 2: 100-Step Deep Enhancement Plan

**Agent Swarm**: Claude Opus (Constitutional), Gemini 3 Pro (Evolutionary), Codex 5.2 (Implementer), GLM-4.7 (Red Team), Superworker (Blue Team), MetaLearner (ICL/CL)

> **Note**: GLM-4.7 accessed via Claude 'glm' alias through Zhipu AI API. Replaces Qwen for Red Team adversarial testing.

**Goal**: Transform ARK into a prescient, metalearning, continuously-improving autonomous VCS

---

## ⚠️ Pre-Production Requirements

**Deploy for internal testing with monitoring. Do not use for production critical repos until:**
1. ✅ S-phase implemented (DONE - LTL integration added)
2. ⬜ Witness cryptographic signing added
3. ✅ Entropy verification from content (DONE - staged file computation)

---

## Phase 0: Immediate Refinements (Before Testing)

### Phase 0.1: Critical (This Session)

| Step | Task | Status | Notes |
|------|------|--------|-------|
| **P0-01** | Add entropy verification from staged files | ✅ DONE | `_compute_entropy_from_staged()` in repository.py |
| **P0-02** | Add empty commit detection | ✅ DONE | `_get_staged_files()` check before Cell Cycle |
| **P0-03** | Add logging to all modules | ✅ DONE | logging_config.py + get_logger() |
| **P0-04** | Implement S-phase with LTL | ✅ DONE | Now calls LTLVerifier |
| **P0-05** | Implement Hysteresis check | ✅ DONE | hysteresis.py created |

### Phase 0.2: Short-Term (This Sprint)

| Step | Task | Status | Notes |
|------|------|--------|-------|
| **P0-06** | Add witness cryptographic signing | ✅ DONE | HMAC-SHA256 in Witness.sign() |
| **P0-07** | Verify witness signatures on read | ✅ DONE | `_verify_witness()` in repository.py |
| **P0-08** | Integrate Hysteresis into Cell Cycle | ✅ DONE | `_check_hysteresis()` in G1 phase |

### Phase 0.3: Medium-Term (Next Sprint)

| Step | Task | Status | Notes |
|------|------|--------|-------|
| **P0-09** | Add Rhizome compaction | ✅ DONE | DGM-compatible soft-compact in `dag.py` |
| **P0-10** | Integrate with LASER entropy profiler | ✅ DONE | 8-dim H* in `integration.py` |
| **P0-11** | Add RER-based inertia scoring | ✅ DONE | `rhizom/ranking.py` - Rhizome Eigenvalue Ranking |

---

## Meta-Objectives

1. **MetaLearning Integration**: Wire ARK to MetaLearner for ICL-based suggestions
2. **Continual Learning (CL)**: Enable ARK to learn from commit patterns over time
3. **In-Context Learning (ICL)**: Use multi-agent context for better synthesis
4. **Prescient Analysis**: Predict commit quality before execution
5. **Deep Testing**: Recursive symbolic/neurosymbolic verification
6. **Performance**: Sub-100ms gate checks, <1s commits
7. **Safety**: Kill-switch integration, spectral stability monitoring

---

## Phase 2.1: MetaLearner Integration (Steps 1-20)

| Step | Task | Agent | Priority |
|------|------|-------|----------|
| **P2-001** | Create `nucleus/ark/meta/` module | Codex | Critical |
| **P2-002** | Implement `MetaLearnerClient` for /suggest API | Codex | Critical |
| **P2-003** | Wire etymology extraction to MetaLearner | Gemini | High |
| **P2-004** | Add commit quality prediction endpoint | Gemini | High |
| **P2-005** | Implement `ark suggest` command | Codex | High |
| **P2-006** | Create CMP prediction from MetaLearner | Gemini | High |
| **P2-007** | Add entropy prediction (pre-commit) | Gemini | High |
| **P2-008** | Implement prescient analysis in S-phase | Gemini | Critical |
| **P2-009** | Wire MetaLearner health to OHM | Codex | Medium |
| **P2-010** | Create feedback loop: commit → outcome → learn | Claude | Critical |
| **P2-011** | Implement experience buffer for ICL | Claude | Critical |
| **P2-012** | Add context window management | Claude | High |
| **P2-013** | Create `ark learn` command for manual learning | Codex | Medium |
| **P2-014** | Implement reward signal from CMP delta | Gemini | High |
| **P2-015** | Add penalty for gate rejections | Gemini | High |
| **P2-016** | Create MetaLearner dashboard widget | Codex | Medium |
| **P2-017** | Implement A2A learning from swarm patterns | Claude | High |
| **P2-018** | Add cross-repo transfer learning hooks | Claude | Medium |
| **P2-019** | Create `ark meta status` command | Codex | Low |
| **P2-020** | Write tests for MetaLearner integration | All | High |

---

## Phase 2.2: Continual Learning Engine (Steps 21-40)

| Step | Task | Agent | Priority |
|------|------|-------|----------|
| **P2-021** | Design CL architecture for ARK | Claude | Critical |
| **P2-022** | Implement experience replay buffer | Codex | Critical |
| **P2-023** | Create pattern memory (successful commits) | Gemini | High |
| **P2-024** | Create anti-pattern memory (rejected commits) | Gemini | High |
| **P2-025** | Implement forgetting mechanism (EWC-style) | Claude | High |
| **P2-026** | Add catastrophic forgetting prevention | Claude | Critical |
| **P2-027** | Create `ark train` command | Codex | Medium |
| **P2-028** | Implement online learning from each commit | Gemini | High |
| **P2-029** | Add batch learning mode | Gemini | Medium |
| **P2-030** | Create learning rate scheduler | Gemini | Medium |
| **P2-031** | Implement curriculum learning for gates | Claude | High |
| **P2-032** | Add difficulty progression logic | Claude | Medium |
| **P2-033** | Create learning metrics dashboard | Codex | Medium |
| **P2-034** | Implement checkpoint saving | Codex | High |
| **P2-035** | Add model versioning in .ark/ | Codex | High |
| **P2-036** | Create `ark checkpoint` command | Codex | Medium |
| **P2-037** | Implement warm-start from checkpoints | Codex | Medium |
| **P2-038** | Add learning history visualization | Codex | Low |
| **P2-039** | Create `ark learning-curve` command | Codex | Low |
| **P2-040** | Write tests for CL engine | All | High |

---

## Phase 2.3: Neural Gate Enhancement (Steps 41-60)

| Step | Task | Agent | Priority |
|------|------|-------|----------|
| **P2-041** | Upgrade NeuralAdapter to use torch | Gemini | Critical |
| **P2-042** | Create learned gate model architecture | Gemini | Critical |
| **P2-043** | Implement feature extraction pipeline | Gemini | High |
| **P2-044** | Add AST embedding for code understanding | Gemini | High |
| **P2-045** | Create commit embedding model | Gemini | High |
| **P2-046** | Implement thrash probability predictor | Gemini | Critical |
| **P2-047** | Add quality score predictor | Gemini | High |
| **P2-048** | Create calibrated confidence intervals | Gemini | High |
| **P2-049** | Implement Bayesian neural gate | Claude | High |
| **P2-050** | Add uncertainty quantification | Claude | High |
| **P2-051** | Create `ark neural status` command | Codex | Medium |
| **P2-052** | Implement neural gate caching | Codex | High |
| **P2-053** | Add GPU acceleration hooks | Codex | Medium |
| **P2-054** | Create distributed inference mode | Codex | Medium |
| **P2-055** | Implement ensemble of neural gates | Gemini | High |
| **P2-056** | Add attention over code changes | Gemini | High |
| **P2-057** | Create interpretability layer | Claude | Medium |
| **P2-058** | Implement saliency maps for gate decisions | Claude | Medium |
| **P2-059** | Add `ark explain` command | Codex | Medium |
| **P2-060** | Write tests for neural gates | All | High |

---

## Phase 2.4: Recursive Symbolic Testing (Steps 61-80)

| Step | Task | Agent | Priority |
|------|------|-------|----------|
| **P2-061** | Create recursive test generator | Codex | Critical |
| **P2-062** | Implement property-based testing | Codex | High |
| **P2-063** | Add mutation testing for gates | GLM | High |
| **P2-064** | Create symbolic execution harness | Claude | High |
| **P2-065** | Implement SMT solver integration | Claude | High |
| **P2-066** | Add constraint satisfaction checks | Claude | High |
| **P2-067** | Create `ark fuzz` command | GLM | Medium |
| **P2-068** | Implement coverage-guided fuzzing | GLM | High |
| **P2-069** | Add adversarial input generation | GLM | Critical |
| **P2-070** | Create security scan integration | GLM | High |
| **P2-071** | Implement formal verification hooks | Claude | High |
| **P2-072** | Add Dafny/F* integration (optional) | Claude | Low |
| **P2-073** | Create `ark verify --deep` command | Codex | High |
| **P2-074** | Implement contract verification | Claude | Medium |
| **P2-075** | Add invariant checker | Claude | Medium |
| **P2-076** | Create regression test generator | Codex | High |
| **P2-077** | Implement test oracle synthesis | Gemini | High |
| **P2-078** | Add golden test management | Codex | Medium |
| **P2-079** | Create `ark golden` command | Codex | Medium |
| **P2-080** | Write tests for testing framework | All | High |

---

## Phase 2.5: Performance & Safety (Steps 81-100)

| Step | Task | Agent | Priority |
|------|------|-------|----------|
| **P2-081** | Profile gate execution time | Codex | Critical |
| **P2-082** | Implement gate parallelization | Codex | High |
| **P2-083** | Add lazy evaluation for unused gates | Codex | High |
| **P2-084** | Create gate result caching | Codex | High |
| **P2-085** | Implement incremental verification | Codex | High |
| **P2-086** | Add bloom filter for duplicate detection | Codex | Medium |
| **P2-087** | Create memory-mapped Rhizome | Codex | High |
| **P2-088** | Implement async commit pipeline | Codex | High |
| **P2-089** | Add connection pooling for MetaLearner | Codex | Medium |
| **P2-090** | Create `ark benchmark` command | Codex | Medium |
| **P2-091** | Integrate kill_switch.py | GLM | Critical |
| **P2-092** | Implement spectral stability monitor | Gemini | Critical |
| **P2-093** | Add anomaly detection | Gemini | High |
| **P2-094** | Create circuit breaker pattern | Codex | High |
| **P2-095** | Implement rate limiting for mutations | GLM | High |
| **P2-096** | Add resource consumption guards | GLM | High |
| **P2-097** | Create `ark safety-check` command | Codex | High |
| **P2-098** | Implement rollback automation | Codex | High |
| **P2-099** | Add telemetry for production monitoring | Codex | Medium |
| **P2-100** | Write final integration tests | All | Critical |

---

## R&D Insights from Subagents

### Claude Opus (Constitutional)
> "The Büchi acceptance condition should be used to verify that CMP improvement *infinitely often* occurs. Consider implementing a Streett automaton for stronger guarantees."

### Gemini 3 Pro (Evolutionary)
> "Thompson Sampling works well for clade selection, but we should explore contextual bandits where the context is the current entropy state. This enables state-dependent selection."

### Codex 5.2 (Implementation)
> "The isogit foundation is solid but consider adding a WASM-compiled Rhizome indexer for browser-native performance. Could enable fully offline ARK operation."

### GLM-4.7 (Red Team)
> "After security audit, recommend: (1) Rate-limit mutation proposals, (2) Add proof-of-work for witness attestation, (3) Implement capability-based access control for Ring 0 operations. Also reviewed RER algorithm - spectral stability checks are critical for preventing ranking manipulation."

### Superworker (Blue Team)
> "Integration priority: (1) OHM health → ARK gates, (2) LASER entropy → H* calculation, (3) MetaLearner → prescient commit quality. These three close 80% of epistemic gaps."

### MetaLearner (Proposed)
> "Add ICL experience retrieval: before each commit, fetch top-5 similar past commits by embedding, include their CMP outcomes in context. This enables few-shot learning for gate decisions."

---

## Success Metrics for Phase 2

| Metric | Target | Current |
|--------|--------|---------|
| Gate execution time | <100ms | ~500ms |
| Commit throughput | >10/min | ~5/min |
| CMP prediction accuracy | >85% | N/A |
| Entropy prediction MAE | <0.1 | N/A |
| Neural gate F1 | >0.9 | 0.75 |
| Test coverage | >90% | 60% |
| Formal spec coverage | >80% | 30% |

---

*Status: PLANNING | Protocol: PBTSO-ARK-P2 | Swarm: 6 Agents*
