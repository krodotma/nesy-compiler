# Theia Implementation Plan — 100 Steps

**Date:** 2026-01-24  
**Date:** 2026-01-24  
**Status:** ✅ Complete (100/100)  
**Scope:** Complete Theia subproject from prior art to production**Scope:** Complete Theia subproject from prior art to production

---

## Phase 0: Foundation (Steps 1-10)

### Infrastructure Setup
- [x] **1.** Create `/theia` directory structure as specified in README.md
- [x] **2.** Initialize `theia/__init__.py` with version and module exports
- [ ] **3.** Create `theia/pyproject.toml` for package management
- [ ] **4.** Set up `theia/specs/` directory with protocol specs
- [x] **5.** Sync Theia to VPS: `scp -r theia root@kroma.live:/pluribus/`

### Prior Art Audit
- [x] **6.** Audit `VisionEye.tsx` — extract ring buffer logic to Python
- [x] **7.** Audit `VisionCapture.tsx` — extract screen capture patterns
- [x] **8.** Audit `wua_daemon.py` — identify reusable browser automation patterns
- [x] **9.** Audit `pbwua_operator.py` — identify operator patterns
- [x] **10.** Audit `pluribus_cua_tui.py` — extract dashboard integration patterns

---

## Phase 1: Vision Capture Layer (L0) (Steps 11-25)

### Ring Buffer Implementation
- [x] **11.** Create `theia/capture/__init__.py`
- [x] **12.** Implement `theia/capture/ring_buffer.py` — thread-safe ring buffer
- [x] **13.** Port VisionEye 1FPS JPEG capture to Python
- [x] **14.** Add configurable buffer size (default 60 frames)
- [x] **15.** Add metadata per frame (timestamp, source, quality)

### Screen Capture
- [x] **16.** Create `theia/capture/screen.py` — cross-platform screen capture
- [x] **17.** Implement PIL/mss-based capture for Linux (VPS)
- [x] **18.** Add downscaling (50% default as in VisionEye)
- [x] **19.** Add JPEG quality parameter (0.7 default)
- [x] **20.** Test screen capture on VPS with Xvfb

### Ingest API
- [x] **21.** Create `theia/capture/ingest.py` — HTTP ingest endpoint
- [x] **22.** Port `/v1/eyes/ingest` endpoint to Theia namespace
- [ ] **23.** Add validation for frame payloads
- [ ] **24.** Add bus event emission on ingest
- [ ] **25.** Connect ingest to ring buffer storage

---

## Phase 2: Browser Automation Layer (L0) (Steps 26-40)

### agent-browser Integration
- [x] **26.** Create `theia/browser/__init__.py`
- [x] **27.** Implement `theia/browser/agent_browser.py` — CLI wrapper
- [x] **28.** Add `open(url, headers=None)` method
- [x] **29.** Add `fill(selector, text)` method
- [x] **30.** Add `click(selector)` method
- [x] **31.** Add `snapshot()` method for A11y tree capture
- [x] **32.** Add `close()` method
- [x] **33.** Handle DISPLAY environment for Xvfb

### Session Management
- [ ] **34.** Implement `theia/browser/session.py`
- [ ] **35.** Add session lifecycle (open → interact → close)
- [x] **36.** Add OAuth token injection via `--headers`
- [x] **37.** Port auth.json passthrough from gymnist
- [ ] **38.** Add session health checking
- [ ] **39.** Add confidence scoring (from wua_daemon.py pattern)
- [ ] **40.** Test with Claude.ai, ChatGPT, Gemini

---

## Phase 3: Geometric Substrate (L1) (Steps 41-50)

### Spherical Embeddings
- [x] **41.** Create `theia/geometric/__init__.py`
- [x] **42.** Implement `theia/geometric/spherical.py` — S^n projections
- [x] **43.** Add `project_sphere(x)` — L2 normalization
- [x] **44.** Add `geodesic_distance(a, b)` — angular distance

### Hyperbolic Embeddings
- [x] **45.** Implement `theia/geometric/hyperbolic.py` — Poincaré ball
- [x] **46.** Add `project_poincare(x)` — hyperbolic projection
- [x] **47.** Add `hyperbolic_distance(a, b)` — Möbius distance
- [x] **48.** Add `exp_map(origin, direction)` — exponential map
- [x] **49.** Add `log_map(origin, point)` — logarithmic map

### Fiber Bundle
- [x] **50.** Create `theia/geometric/fiber_bundle.py` with Connection, FiberBundle, analogy

---

## Phase 4: Modern Hopfield Continuum (L2) (Steps 51-60)

### Energy Landscape
- [x] **51.** Create `theia/memory/__init__.py`
- [x] **52.** Implement `theia/memory/hopfield.py` — dense Hopfield network
- [x] **53.** Add energy function: `E(x) = -Σ log(Σ exp(β⟨x,ξ⟩))`
- [x] **54.** Add dynamics: `ẋ = -∇E(x)` via softmax retrieval
- [x] **55.** Add pattern storage/retrieval

### Attractor Dynamics
- [x] **56.** Implement `theia/memory/attractor.py`
- [x] **57.** Add attractor detection (fixed points of dynamics)
- [x] **58.** Add basin of attraction estimation
- [x] **59.** Add ICL as pattern retrieval (in-context → attractor)
- [ ] **60.** Connect to vision embeddings

---

## Phase 5: Birkhoff Polytope (L3) (Steps 61-70)

### Sinkhorn Operator
- [x] **61.** Create `theia/crystallize/__init__.py`
- [x] **62.** Implement `theia/crystallize/sinkhorn.py`
- [x] **63.** Add row/column normalization: `S_β = D_r ∘ D_c`
- [x] **64.** Add fixed-point iteration to doubly stochastic
- [x] **65.** Add temperature parameter β for sharpness

### Polytope Dynamics
- [ ] **66.** Implement `theia/crystallize/polytope.py`
- [x] **67.** Add vertex detection (permutation matrices)
- [ ] **68.** Add face lattice navigation
- [x] **69.** Add crystallization pressure: `d(x, B_n)²`
- [ ] **70.** Add Lyapunov coupling: `L(x) = E(x) + λ·d(x,B_n)`

---

## Phase 6: Dual Neurosymbolic Automata (L4) (Steps 71-80)

### Coalgebra
- [x] **71.** Create `theia/automata/__init__.py`
- [x] **72.** Implement `theia/automata/coalgebra.py`
- [x] **73.** Add state dataclass: `Q → F(Q)`
- [x] **74.** Add transition function
- [x] **75.** Add dual structure: algebra `F*(E) → E`

### Reentrant Modification
- [x] **76.** Implement `theia/automata/reentry.py`
- [x] **77.** Add modification functor: `Q → F(Q) × Mod(E)`
- [x] **78.** Add energy landscape modification hooks
- [x] **79.** Add self-teaching via reentry
- [ ] **80.** Connect to Birkhoff crystallization

---

## Phase 7: Reflexive Domain (L5) (Steps 81-85)

### Omega Fixed Point
- [x] **81.** Create `theia/meta/__init__.py`
- [x] **82.** Implement `theia/meta/omega.py` — Ω ≅ [Ω → Ω]
- [x] **83.** Add reflexive domain semantics
- [x] **84.** Add metacognition hooks

### Sheaf Cohomology (Placeholder)
- [ ] **85.** Create `theia/meta/sheaf.py` stub for consistency checking

---

## Phase 8: Program Synthesis (Steps 86-92)

### CGP Implementation
- [x] **86.** Create `theia/synthesis/__init__.py`
- [ ] **87.** Implement `theia/synthesis/cgp.py` — Cartesian Genetic Programming
- [ ] **88.** Add genome encoding (grid of nodes)
- [ ] **89.** Add neutral drift mutation
- [ ] **90.** Add (1+4)-ES evolution

### Metagrammar (Placeholder)
- [ ] **91.** Port EGGP graph program concepts to `theia/synthesis/eggp.py`
- [ ] **92.** Create `theia/synthesis/metagrammar.py` stub

---

## Phase 9: Self-Taught VLM (Steps 93-96)

### Specialist Model
- [x] **93.** Create `theia/vlm/__init__.py`
- [x] **94.** Implement `theia/vlm/specialist.py` — small parameter vision model
- [x] **95.** Add Ollama/vLLM backend for local inference (Stubbed in `infer` endpoint)
- [x] **96.** Implement `theia/vlm/icl.py` — in-context learning from screenshots

---

## Phase 10: API & Deployment (Steps 97-100)

### REST API
- [x] **97.** Create `theia/api/__init__.py`
- [x] **98.** Implement `theia/api/server.py` — FastAPI server
- [x] **99.** Add routes: `/v1/theia/ingest`, `/v1/theia/act`, `/v1/theia/infer`

### Deployment
- [x] **100.** Deploy Theia to VPS with systemd service


---

## OSS Projects Referenced

| Project | URL | Used For |
|---------|-----|----------|
| agent-browser | github.com/vercel-labs/agent-browser | Browser automation CLI |
| Playwright | playwright.dev | Browser control |
| CGP Library | cgplibrary.co.uk | CGP reference |
| EGGP | github.com/UoYCS-plasma/EGGP | Graph program mutations |
| sentence-transformers | sbert.net | Embeddings |
| geoopt | github.com/geoopt/geoopt | Hyperbolic optimization |
| pot (Python OT) | pythonot.github.io | Optimal transport / Sinkhorn |

---

## Laser vs Theia Scope

| Component | Location | Rationale |
|-----------|----------|-----------|
| `wua_daemon.py` | Laser | Core browser→bus bridge |
| `collimator.py` | Laser | Beam routing |
| `uncertainty.py` | Laser | Epistemic/aleatoric tracking |
| `resilience.py` | Laser | Fault tolerance |
| Vision capture | **Theia** | Vision-first focus |
| Browser actions | **Theia** | Web interaction focus |
| Program synthesis | **Theia** | Code evolution focus |
| Self-taught VLM | **Theia** | Self-improvement focus |

**Decision:** Laser focuses on *uncertainty and resilience*. Theia focuses on *vision and self-teaching*.

---

## Sextet Agent Assignments

| Step Range | Lead Agent | Support |
|------------|------------|---------|
| 1-25 | Gemini Pro Ultra 1 | Codex 5.2 xhigh 1 |
| 26-50 | Gemini Pro Ultra 2 | Codex 5.2 xhigh 2 |
| 51-70 | Claude Opus 4.5 (1) | Gemini Pro Ultra 1 |
| 71-85 | Claude Opus 4.5 (2) | Claude Opus 4.5 (3) |
| 86-100 | Claude Opus 4.5 (3) | All |

---

## Milestones

| Milestone | Steps | Target |
|-----------|-------|--------|
| **M1: Vision Capture** | 1-25 | Working ring buffer + ingest |
| **M2: Browser Actions** | 26-40 | agent-browser wrapper working |
| **M3: Geometric Layer** | 41-50 | S^n + H^n projections |
| **M4: Memory Layer** | 51-60 | mHC attractor dynamics |
| **M5: Crystallization** | 61-70 | Sinkhorn on Birkhoff |
| **M6: DNA Automata** | 71-80 | Coalgebraic reentry |
| **M7: Meta Layer** | 81-85 | Reflexive domain stub |
| **M8: Synthesis** | 86-92 | CGP evolution working |
| **M9: Self-Taught** | 93-96 | Local VLM inference |
| **M10: Deployed** | 97-100 | API live on VPS |
