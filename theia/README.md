# Theia — Vision-First Neurosymbolic Agent

**Titaness of Sight: Endowing AI with celestial vision.**

> *"Theia (or Thia) — the Greek goddess of sight, vision, and the shimmering light of the sky. Mother of Helios (Sun), Selene (Moon), and Eos (Dawn). She endows gold, silver, and gems with their sparkle."*

---

## Purpose

Theia is a **vision-first neurosymbolic agent** that consolidates all Pluribus vision, browser automation, program synthesis, and self-teaching capabilities into a unified subproject.

### Core Capabilities

| Capability | Implementation |
|------------|----------------|
| **Browser Automation** | agent-browser + browser_session_daemon |
| **Screen Vision** | VisionEye ring buffer capture |
| **Self-Teaching** | Small parameter specialist model with ICL |
| **Program Synthesis** | CGP + EGGP metagrammar / AST grammars |
| **OAuth WebChat** | Gymnist → Theia passthrough |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         THEIA STACK                             │
├─────────────────────────────────────────────────────────────────┤
│ L5: Ω Reflexive Domain                                          │
│     Self-modeling, metacognition, fixed-point semantics         │
├─────────────────────────────────────────────────────────────────┤
│ L4: DNA (Dual Neurosymbolic Automata)                           │
│     Coalgebraic state → F(state) × Mod(E)                       │
│     Reentrant self-modification                                 │
├─────────────────────────────────────────────────────────────────┤
│ L3: Birkhoff Polytope Dynamics                                  │
│     Sinkhorn crystallization to discrete structures             │
│     Face lattice = partial symbolic commitments                 │
├─────────────────────────────────────────────────────────────────┤
│ L2: Modern Hopfield Continuum (mHC)                             │
│     Energy landscape, attractor dynamics, attention             │
│     Exponential capacity in dimension                           │
├─────────────────────────────────────────────────────────────────┤
│ L1: Geometric Substrate (S^n ⊣ H^n)                             │
│     Spherical (local similarity) + Hyperbolic (hierarchy)       │
│     Fiber bundle with learned connection/curvature              │
├─────────────────────────────────────────────────────────────────┤
│ L0: Vision Capture + Browser Automation                         │
│     agent-browser, VisionEye, screen recording                  │
│     /v1/eyes/ingest endpoint                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prior Art Integration

### From Laser (stays in Laser)
- `wua_daemon.py` — Browser → bus bridge with confidence scoring
- `collimator.py` — Beam focusing / routing
- `uncertainty.py` — Aleatoric/epistemic tracking

### From WUA (migrates to Theia)
- `pbwua_operator.py` → `theia/operators/browser.py`
- `wua_protocol_v1.md` → `theia/specs/vision_protocol.md`

### From Vision Components (migrates to Theia)
- `VisionEye.tsx` → Reference for `theia/capture/ring_buffer.py`
- `VisionCapture.tsx` → Reference for capture UI

### From SOTA Research (informs Theia design)
- `graph_evolutionary_programming_distillation.md` — CGP/EGGP/MAGE
- `birkhoff-neurosymbolic-crystals.md` — vec2vec→mHC→Birkhoff→DNA

---

## Sextet Agent Flow

Theia uses a **sextet** of agents for its core API flow:

| Role | Model | Purpose |
|------|-------|---------|
| **Planner 1** | Claude Opus 4.5 (ultrathink) | Strategic planning, architecture |
| **Planner 2** | Claude Opus 4.5 (ultrathink) | Validation, adversarial review |
| **Planner 3** | Claude Opus 4.5 (ultrathink) | Synthesis, final plan |
| **Executor 1** | Gemini Pro Ultra | High-complexity implementation |
| **Executor 2** | Gemini Pro Ultra | Parallel implementation path |
| **Verifier 1** | Codex 5.2 xhigh | Code verification, testing |
| **Verifier 2** | Codex 5.2 xhigh | Integration, deployment |

```
Sextet Flow:
  [P1]──┬──▶[E1]──┬──▶[V1]
        │         │
  [P2]──┼──▶[E2]──┼──▶[V2]
        │         │
  [P3]──┴─────────┴──────▶ Final Output
```

---

## API Surface

### `/v1/theia/ingest` — Vision Capture
```python
POST /v1/theia/ingest
{
  "frames": ["base64...", ...],  # Ring buffer frames
  "timestamp": 1735170000,
  "meta": {"fps": 1, "source": "screen"}
}
```

### `/v1/theia/act` — Browser Action
```python
POST /v1/theia/act
{
  "action": "fill",
  "selector": "textarea",
  "value": "Hello world",
  "provider": "claude-ab"
}
```

### `/v1/theia/infer` — VLM Inference
```python
POST /v1/theia/infer
{
  "image": "base64...",
  "prompt": "What do you see?",
  "model": "theia-local"  # Self-taught specialist
}
```

---

## Directory Structure

```
theia/
├── README.md           # This file
├── __init__.py
├── capture/            # Vision capture (L0)
│   ├── ring_buffer.py
│   ├── screen.py
│   └── ingest.py
├── browser/            # Browser automation (L0)
│   ├── agent_browser.py   # Wraps agent-browser CLI
│   └── session.py
├── geometric/          # Geometric substrate (L1)
│   ├── spherical.py
│   ├── hyperbolic.py
│   └── fiber_bundle.py
├── memory/             # mHC (L2)
│   ├── hopfield.py
│   └── attractor.py
├── crystallize/        # Birkhoff (L3)
│   ├── sinkhorn.py
│   └── polytope.py
├── automata/           # DNA (L4)
│   ├── coalgebra.py
│   └── reentry.py
├── meta/               # Reflexive (L5)
│   ├── omega.py
│   └── sheaf.py
├── synthesis/          # Program synthesis
│   ├── cgp.py
│   ├── eggp.py
│   └── metagrammar.py
├── vlm/                # Self-taught vision model
│   ├── specialist.py
│   └── icl.py
├── api/                # REST API
│   ├── server.py
│   └── routes.py
└── specs/
    ├── vision_protocol.md
    └── sextet_flow.md
```

---

## Comparison: agent-browser vs browser_session_daemon

| Feature | agent-browser (Vercel) | browser_session_daemon |
|---------|------------------------|------------------------|
| **Architecture** | Rust CLI + Node.js daemon | Pure Python + Playwright |
| **Startup** | Fast (native binary) | Slower (Python import) |
| **Auth** | `--headers` Bearer token | VNC login + cookie storage |
| **Output** | `snapshot` (A11y tree) | Full DOM capture |
| **Persistence** | Session-based | Full profile persistence |
| **Lines of Code** | ~2000 (estimate) | 4518 lines |
| **Dependencies** | npm install | pip install + playwright |

**Recommendation**: Use agent-browser for lightweight OAuth webchat, browser_session_daemon for persistent VNC-backed sessions.

---

## Quick Start

```bash
# Install agent-browser
npm i -g agent-browser
agent-browser install

# Test capture
agent-browser open https://claude.ai
agent-browser snapshot > claude_snapshot.txt

# Run Theia API
python -m theia.api.server
```

---

## Related Projects

- **Laser** — Beam routing, uncertainty, resilience (Theia focuses on vision)
- **Gymnist** — Renamed to Theia for vision-first agents
- **Membrane/Agent0** — Training infrastructure for Theia's self-taught models
