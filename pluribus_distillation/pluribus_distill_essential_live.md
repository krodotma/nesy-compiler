# Pluribus Essential Live Distillation (PBTSO / SemOps Aligned)

Date: 2026-01-23
Scope: Implemented and traceable components only. The bus is evidence; PBTSO tmux is the control plane.
Sources: repo files + /Users/kroma/pluribus_distillation/distill_report_super.md

---

## 1) Control Plane (PBTSO Canonical)

PBTSO is the unified orchestration control plane. It is active, not passive. The tmux session is authoritative; bus topics are evidence only.

Key files:
- /Users/kroma/pluribus_evolution/nucleus/specs/pbtso_protocol_v1.md
- /Users/kroma/pluribus_evolution/nucleus/specs/semops.json
- /Users/kroma/pluribus_evolution/nucleus/tools/tmux_swarm_orchestrator.py
- /Users/kroma/pluribus_evolution/nucleus/tools/tbtso_a2a.py
- /Users/kroma/pluribus_evolution/nucleus/tools/pbtso_task_daemon.py
- /Users/kroma/pluribus_evolution/nucleus/tools/pblanes_operator.py
- /Users/kroma/pluribus_evolution/nucleus/tools/ckin_report.py
- /Users/kroma/pluribus_evolution/nucleus/tools/oiterate_operator.py

PBTSO operational summary:
- Spawn/loop/recruit via tmux_swarm_orchestrator (PBTSO / PBTSWARM / PBLOOP).
- A2A mutual inference via tbtso_a2a.py (dual emit tbtso.* and pbtso.*).
- Task ingress via pbtso_task_daemon.py (task.create -> task_ledger + pbtso.task.created).
- Lane visibility via pblanes_operator.py (reads nucleus/state/lanes.json).

---

## 2) Live System Topology (Primary + Evolution + Governance)

### 2.1 Primary Trunk (Execution)

- World Router: /Users/kroma/pluribus/nucleus/tools/world_router.py
  - Unified gateway for inference, CUA, VNC, storage, identity, bus stream endpoints.
- Omega Dispatcher: /Users/kroma/pluribus/omega_dispatcher.py
  - Manifest-driven routing to domains/apps.
- Agent Bus (primary impl): /Users/kroma/pluribus/nucleus/tools/agent_bus.py
  - File-backed NDJSON bus (Veteran Protocol v2).
- OHM (primary impl): /Users/kroma/pluribus/nucleus/tools/ohm.py
  - Omega health and task tracking; emits ohm.status/ohm.alert.
- Auralux pipeline: /Users/kroma/pluribus/nucleus/auralux/pipeline_orchestrator.ts
  - Audio VAD -> SSL -> Vocoder pipeline orchestrator.
- Meta Learner: /Users/kroma/pluribus/meta_learner/learner.py
  - Experience buffer + registry (lightweight RL loop).

### 2.2 Evolution Trunk (Observer/Refiner/Synthesizer)

- Observer daemon: /Users/kroma/pluribus_evolution/pluribus_evolution/observer/main.py
  - Emits evolution.observer.analysis/drift/vector events.
- Observer components:
  - /Users/kroma/pluribus_evolution/pluribus_evolution/observer/code_analyzer.py
  - /Users/kroma/pluribus_evolution/pluribus_evolution/observer/drift_detector.py
  - /Users/kroma/pluribus_evolution/pluribus_evolution/observer/vector_profiler.py
- Refiner: /Users/kroma/pluribus_evolution/pluribus_evolution/refiner/proposal_generator.py
  - Emits evolution.refiner.proposal batches.
- Synthesizer: /Users/kroma/pluribus_evolution/pluribus_evolution/synthesizer/patch_generator.py
  - Emits evolution.synthesizer.patch artifacts.
- Bridge: /Users/kroma/pluribus_evolution/pluribus_evolution/bridge/bus_mirror.py
  - Mirrors primary bus into evolution bus and emits evolution.mirror.*.

### 2.3 LASER/LENS Synthesis

- Collimator: /Users/kroma/pluribus_evolution/laser/collimator.py
- Entropy profiler: /Users/kroma/pluribus_evolution/laser/entropy_profiler.py
- Synthesizer: /Users/kroma/pluribus_evolution/laser/synthesizer.py
- Uncertainty: /Users/kroma/pluribus_evolution/laser/uncertainty.py

### 2.4 Governance / Oversight

- Antigravity meta-tool: /Users/kroma/.gemini/antigravity/
  - code_tracker: /Users/kroma/.gemini/antigravity/code_tracker/active/
  - knowledge base: /Users/kroma/.gemini/antigravity/knowledge/pluribus_evolution/
- OHM (evolution impl): /Users/kroma/pluribus_evolution/nucleus/tools/ohm.py

---

## 3) Runtime Flows (Traceable)

### 3.1 PBTSO Task Ingress (Dialogos -> Ledger)

Flow:
1) UI dispatch: /Users/kroma/pluribus_evolution/nucleus/dashboard/src/components/dialogos/logic/PBTSOBridge.ts
   - Emits task.create with correlation_id.
2) Ingress daemon: /Users/kroma/pluribus_evolution/nucleus/tools/pbtso_task_daemon.py
   - Reads task.create, appends to task_ledger, emits pbtso.task.created and legacy tbtso.task.created.
3) UI ack: PBTSOBridge subscribes to pbtso.task.created/tbtso.task.created and marks atom actualized.

Evidence topics:
- task.create
- pbtso.task.created
- tbtso.task.created

### 3.2 PBTSO A2A Coordination (Swarm Control)

Flow:
1) Orchestrator spawns swarm: /Users/kroma/pluribus_evolution/nucleus/tools/tmux_swarm_orchestrator.py
2) A2A coordination: /Users/kroma/pluribus_evolution/nucleus/tools/tbtso_a2a.py
   - Emits tbtso.a2a.* and pbtso.a2a.* (mirrored), plus a2a.handshake.propose and a2a.heartbeat.

Evidence topics (canonical + legacy):
- pbtso.a2a.swarm.init / tbtso.a2a.swarm.init
- pbtso.a2a.swarm.heartbeat / tbtso.a2a.swarm.heartbeat
- pbtso.a2a.swarm.complete / tbtso.a2a.swarm.complete
- pbtso.a2a.lane.progress / tbtso.a2a.lane.progress
- a2a.handshake.propose
- a2a.heartbeat

### 3.3 Evolution Observer Loop

Flow:
1) systemd service: /Users/kroma/pluribus_evolution/nucleus/deploy/systemd/pluribus-evolution.service
   - ExecStart: /usr/bin/python3 -m pluribus_evolution.observer.main
2) Observer daemon: /Users/kroma/pluribus_evolution/pluribus_evolution/observer/main.py
   - Emits evolution.observer.analysis, evolution.observer.drift, evolution.observer.vector.
3) Bridge mirror: /Users/kroma/pluribus_evolution/pluribus_evolution/bridge/bus_mirror.py
   - Mirrors primary bus to evolution bus and emits evolution.mirror.*.

Evidence topics:
- evolution.observer.analysis
- evolution.observer.drift
- evolution.observer.vector
- evolution.mirror.*

### 3.4 LASER/LENS Synthesis Path

Flow:
1) Collimator (routing, lane selection): /Users/kroma/pluribus_evolution/laser/collimator.py
2) Entropy profiling: /Users/kroma/pluribus_evolution/laser/entropy_profiler.py
3) Multi-model synthesis: /Users/kroma/pluribus_evolution/laser/synthesizer.py
4) Uncertainty scoring: /Users/kroma/pluribus_evolution/laser/uncertainty.py

### 3.5 OHM Monitoring and Gating

Flow:
1) OHM monitors bus + task ledger: /Users/kroma/pluribus_evolution/nucleus/tools/ohm.py and /Users/kroma/pluribus/nucleus/tools/ohm.py
2) Emits ohm.status / ohm.alert and omega.* metrics (see OMEGA_TOPICS in OHM).

Evidence topics:
- ohm.status
- ohm.alert
- omega.heartbeat
- omega.queue.depth
- omega.providers.health
- omega.dispatch.tick

---

## 4) Implemented vs Theoretical (Reconciled)

Confirmed implemented (traceable in repo):
- PBTSO control plane (tmux_swarm_orchestrator.py)
- PBTSO task ingress (pbtso_task_daemon.py)
- A2A coordination (tbtso_a2a.py)
- Lanes reporting (pblanes_operator.py)
- CKIN reporting (ckin_report.py)
- OITERATE loop (oiterate_operator.py)
- Evolution observer/refiner/synthesizer pipeline
- LASER/LENS modules
- OHM monitoring

Not evidenced as implemented (from super report + file checks):
- rhizome_sync.py referenced in prior reports but not found in repo
- CMP++ upgrades (entelexis proposals) are not verified as implemented
- WUA listed as planned; no runtime service evidence in current trace
- Subtree extraction plans (LASER/PORTAL) are documented, not confirmed as executed

---

## 5) Minimal Live Path (Bare-Metal, Essential)

If the goal is the bare minimum live system with PBTSO control plane and evolution loop:

1) PBTSO control plane:
   - /Users/kroma/pluribus_evolution/nucleus/tools/tmux_swarm_orchestrator.py
   - /Users/kroma/pluribus_evolution/nucleus/tools/tbtso_a2a.py
   - /Users/kroma/pluribus_evolution/nucleus/tools/pbtso_task_daemon.py
2) Evidence bus:
   - /Users/kroma/pluribus_evolution/nucleus/tools/agent_bus.py (FalkorDB + NDJSON)
3) Evolution observer loop:
   - /Users/kroma/pluribus_evolution/pluribus_evolution/observer/main.py
   - /Users/kroma/pluribus_evolution/nucleus/deploy/systemd/pluribus-evolution.service
4) OHM gating:
   - /Users/kroma/pluribus_evolution/nucleus/tools/ohm.py

---

## 6) Reference: PBTSO SemOps Anchors

Canonical operators used in this report (registry in semops.json):
- PBTSO / PBTSWARM / PBLOOP
- PBTSO_A2A (legacy filename tbtso_a2a.py)
- PBLANES
- CKIN
- OITERATE

---

End of report.
