# Pluribus Lexicon: Formal Vocabulary, Grammar, and Ontological Priors

> **CANONICAL SOURCE SUPERSEDED (Semioticalysis)**
> 
> As of Sagent Iteration 5, the definitions for **Semioticalysis**, **PLURIBUS Modality**, and **SemIps** 
> are authoritatively defined in [`nucleus/docs/concepts/PLURIBUS_OPERATOR.md`](../docs/concepts/PLURIBUS_OPERATOR.md).
> This document remains canonical for general vocabulary, routing, and idiolect.

> A living specification of the pluribus idiolect for cognitive design patterns,
> agent coordination, and evolutionary computation.

---

## 1. Foundational Ontology

### 1.1 Agents and Identities

| Term | Definition | Grammar |
|------|------------|---------|
| **Agent** | An autonomous computational entity capable of inference, action, and coordination | `agent := {id, type, persona, capabilities}` |
| **Ring** | Security/capability tier (Ring 0 = architect, Ring 3 = ephemeral worker) | `ring := 0 | 1 | 2 | 3` |
| **Persona** | Behavioral profile injected into agent context | `persona := {id, voice, constraints, expertise}` |
| **Actor** | The identity under which actions are attributed (audit trail) | `actor := string` |
| **CAGENT** | Citizen agent: protocol-compliant per CAGENT spec; classed as superagent (full bootstrap) or superworker (limited bootstrap). Unqualified “CAGENT” defaults to the superworker tier unless explicitly marked SAGENT. | `cagent := {class: superagent | superworker, tier: full | limited, actor, persona?, scope?}` |
| **SAGENT** | Superagent citizen: a CAGENT with full bootstrap + full repo context; explicit term for superagent-tier work | `sagent := cagent{class=superagent, tier=full}` |
| **Superagent** | Full-citizenship CAGENT capable of orchestration, planning, and subagent spawning (alias: SAGENT) | `superagent := cagent{class=superagent}` |
| **Superworker** | Limited-citizenship CAGENT specialized to narrow tasks; verification + guardrails, no full repo bootstrap, no subagent spawning | `superworker := cagent{class=superworker}` |
| **PBSAGENT** | Pluribus Superagent: a CAGENT superagent (manager/director/planner/task supervisor) with native CLI, frontier scope, and full repo knowledge | `pbsagent := superagent{role=manager}` |

### 1.2 Agent Types

```
agent_type :=
  | ring0.architect     -- Full project understanding, security-critical
  | ring1.coordinator   -- Orchestrates subagents, topology decisions
  | ring2.specialist    -- Domain expertise (code, research, creative)
  | ring3.worker        -- Ephemeral, narrow task execution
  | subagent.narrow_coder
  | subagent.researcher
  | subagent.auditor
```

### 1.3 Flows (Agent Coordination Topology)

| Flow Type | Definition | Example |
|-----------|------------|---------|
| **intra** | Same agent type, same instance | Claude subagent → Claude subagent |
| **inter** | Different pluribus agents | Claude ↔ Codex |
| **extra** | External systems | Pluribus → Gemini API, Pluribus → ChatGPT Web |

```
flow := {
  intra: [agent_id...],
  inter: [agent_id...],
  extra: [provider_id...]
}
```

---

### 1.4 Primary Modes (Mode Lattice)

These modes are the canonical operational vocabulary across Pluribus and holons. They overlay existing pipelines (curate -> distill -> hypothesize -> apply -> verify) and must not be replaced by new synonyms.

| Term | Definition | Grammar |
|------|------------|---------|
| **Ingest** | Capture raw signals/artifacts and emit evidence; context governance curates scope and avoids poisoning; static assets only in transfiguration flow | `ingest := {source, payload, evidence, context_governance}` |
| **Inception** | Canonicalize ingress and assign lineage/ids before fork | `inception := {ingress_id, lineage_id, schema_valid}` |
| **Delineate** | Apply gates and boundaries; decide AM/SM/decay | `delineate := {sextet, omega, verdict}` |
| **Orchestrate** | Coordinate agents/steps and handoffs | `orchestrate := {route_plan, topology, fanout}` |
| **Align** | Verify goals against etymons + active goals | `align := {goal, etymon, confidence}` |
| **Operationalize** | Package into holon/commit/service and verify | `operationalize := {artifact, verification, deployment}` |

---

**Note:** Context governance is an Ingest property, not a new primary mode.

```
context_governance := {scope, filters, dedupe, provenance_checks, window_budget}
```

## 2. Inference and Routing

### 2.1 Lens/Collimator Grammar

The Lens classifies queries; the Collimator routes them.

```
LensRequest := {
  req_id: uuid,
  goal: string,
  kind: request_kind,
  effects: effect_class,
  prefer: {providers: [provider...], require_model_prefix: string?}
}

RoutePlan := {
  req_id: uuid,
  depth: depth_class,
  lane: lane_class,
  provider: provider_id,
  context_mode: context_class,
  topology: topology_class,
  fanout: nat,
  persona_id: persona_id
}
```

### 2.2 Depth Classification

| Depth | Definition | Resource Allocation |
|-------|------------|---------------------|
| **narrow** | Isolated task, minimal context needed | Lighter agents, faster timeout |
| **deep** | Project-aware, requires full understanding | Top-tier agents, extended context |

```
depth_class := narrow | deep

depth_heuristics:
  deep ← kind ∈ {audit, benchmark}
  deep ← goal contains {architecture, design, spec, protocol, schema}
  deep ← len(goal) > 240
  narrow ← otherwise
```

### 2.3 Lane Classification

| Lane | Purpose | Use Case |
|------|---------|----------|
| **dialogos** | Streaming conversational UX | Interactive chat, human-in-loop |
| **pbpair** | Structured paired model consult | Deep analysis, verification |
| **strp** | Async task queue | Background distillation, batch |

```
lane_class := dialogos | pbpair | strp

lane_selection:
  pbpair ← depth = deep
  dialogos ← depth = narrow ∧ effects = none
  strp ← async ∨ topology ∈ {star, peer_debate}
```

### 2.4 Topology

| Topology | Structure | Coordination |
|----------|-----------|--------------|
| **single** | One agent | Direct execution |
| **star** | 1 coordinator + N workers | Hub orchestrates fanout |
| **peer_debate** | N parallel peers | Consensus or best-of-N |

```
topology_class := single | star | peer_debate

topology := {
  topology: topology_class,
  fanout: nat,              -- Number of agents
  coord_budget_tokens: nat  -- Token budget for coordination overhead
}
```

### 2.5 Context Mode

| Mode | Context Window | Use Case |
|------|----------------|----------|
| **min** | Minimal, task-only | Isolated narrow queries |
| **lite** | Light project context | Moderate complexity |
| **full** | Complete project state | Deep architectural work |

```
context_class := min | lite | full
```

---

## 3. Cognitive Design Patterns

### 3.1 Uncertainty Classes

| Class | Definition | Resolution Strategy |
|-------|------------|---------------------|
| **aleatoric** | Irreducible randomness (inherent to domain) | Sample multiple, aggregate |
| **epistemic** | Reducible by gathering more information | Research, retrieval, clarification |

```
uncertainty := {
  type: aleatoric | epistemic,
  source: string,
  confidence: float[0,1],
  resolution_strategy: strategy
}

strategy :=
  | sample_aggregate(n: nat)           -- Aleatoric: multiple samples
  | retrieve(sources: [source...])     -- Epistemic: gather info
  | clarify(question: string)          -- Epistemic: ask human
  | defer(to: agent_id)                -- Delegate to specialist
```

### 3.2 Gap Filling Patterns

```
gap_pattern :=
  | aleatoric_sampling   -- Generate N candidates, select best
  | epistemic_retrieval  -- RAG/KG lookup to fill knowledge gap
  | epistemic_clarify    -- Human-in-loop question
  | dialectic_synthesis  -- Thesis/antithesis → synthesis
  | evolutionary_search  -- VGT/HGT mutation + selection
```

### 3.3 Dialog Patterns

| Pattern | Structure | Use Case |
|---------|-----------|----------|
| **monologic** | Single agent response | Simple queries |
| **dialogic** | Agent ↔ Human turn-taking | Interactive refinement |
| **polylogic** | Multi-agent discussion | Complex problem decomposition |
| **dialectic** | Thesis → Antithesis → Synthesis | Conflict resolution |

```
dialog_pattern := monologic | dialogic | polylogic | dialectic

dialog_session := {
  pattern: dialog_pattern,
  participants: [agent_id | "human"],
  turns: [turn...],
  goal: string,
  termination: termination_condition
}

turn := {
  speaker: agent_id | "human",
  content: string,
  intent: intent_class,
  effects: [effect...]
}

intent_class :=
  | assert      -- State a claim
  | query       -- Request information
  | propose     -- Suggest action/direction
  | challenge   -- Question/dispute
  | synthesize  -- Combine/reconcile
  | commit      -- Agree/accept
```

### 3.4 Supervision Modes

```
supervision_mode :=
  | human_in_loop    -- Every action requires approval
  | human_on_loop    -- Human monitors, can intervene
  | unsupervised     -- Autonomous within policy bounds
  | hybrid(policy)   -- Conditional based on risk/impact
```

---

## 4. Evolutionary Computation

### 4.1 Gene Transfer

| Type | Definition | Scope |
|------|------------|-------|
| **VGT** | Vertical Gene Transfer | Parent → Child (inheritance) |
| **HGT** | Horizontal Gene Transfer | Peer → Peer (cross-pollination) |

```
gene := {
  id: hash,
  content: artifact,
  lineage: [ancestor_hash...],
  fitness: float,
  provenance: {author, timestamp, source}
}

vgt_transfer := parent_gene → child_gene  -- Inheritance
hgt_transfer := peer_gene → recipient_gene  -- Splice with guard
```

### 4.2 Lineage Tracking

```
lineage := {
  luca: hash,           -- Last Universal Common Ancestor
  branch: string,       -- Current evolutionary branch
  generation: nat,      -- Distance from LUCA
  mutations: [mutation...]
}

mutation := {
  type: point | splice | delete | duplicate,
  location: path,
  delta: diff,
  fitness_delta: float
}
```

### 4.3 Selection Pressure

```
fitness_function := artifact → float[0,1]

selection := {
  strategy: tournament | roulette | elitist | novelty,
  population_size: nat,
  survival_rate: float,
  mutation_rate: float
}
```

### 4.4 Semantic Naming (HOLON)

```
semantic_branch :=
  "evo/" date "-" holon "." domain "." intent "." surface "." store "." phase "." actor "." ark

semantic_agent_id :=
  class "." role "." domain "." focus "." lane ".r" ring "." model "." variant
```

- **Purpose:** embed intent, scope, and lineage into identifiers for IRKG parsing.
- **Spec:** `nucleus/specs/holon_semantic_naming_v1.md`

---

## 5. Provider and Session Grammar

### 5.1 Provider Types

```
provider_class :=
  | api          -- Direct API call (Anthropic, OpenAI, Vertex)
  | cli          -- CLI tool (claude, codex, gemini)
  | web_session  -- Authenticated browser OAuth session
  | mock         -- Deterministic test responses

provider := {
  id: provider_id,
  class: provider_class,
  model: model_id,
  available: bool,
  cooldown_until: timestamp?,
  auth: auth_state
}
```

### 5.2 Web Session Routing

```
web_session := {
  provider: gemini-web | claude-web | chatgpt-web,
  endpoint: url,
  auth_method: oauth | cookie | token,
  session_valid: bool,
  fallback: api_provider
}
```

### 5.3 Fallback Chain

```
fallback_chain := [provider_id...]

fallback_policy := {
  order: fallback_chain,
  on_failure: skip | retry(n) | abort,
  cooldown_s: nat,
  emit_incident: bool
}
```

---

## 6. Bus and Evidence

### 6.1 Event Grammar

```
bus_event := {
  id: uuid,
  ts: timestamp,
  iso: iso8601,
  topic: topic_path,
  kind: event_kind,
  level: log_level,
  actor: actor_id,
  data: json
}

event_kind := request | response | artifact | metric | log
log_level := debug | info | warn | error

topic_path := segment ("." segment)*
segment := [a-z0-9_]+
```

### 6.2 Semantic Events

```
semantic_event := bus_event & {
  semantic: string,        -- Human-readable summary
  reasoning: string?,      -- Why this happened
  actionable: [string...], -- Suggested next steps
  impact: low | medium | high
}
```

### 6.2.1 Multi-Agent Trace Extension (MATM)

Bus events MAY include trace metadata for artifact preservation and cross-agent correlation:

```
trace_extension := {
  session_id: uuid,                              -- Agent session identifier
  parent_session_id: uuid?,                      -- Parent if spawned subagent
  dialogos_cell_id: dialogos.cell.<s>.<t>.<seq>, -- Conversation cell reference
  hexis_checkpoint: msg_id?,                     -- Last consumed hexis message
  lineage_hash: sha256[:16]?,                    -- Evolutionary ancestry (VGT/HGT)
  entelexis_goal: string?,                       -- Active purpose/telos
  correlation_id: uuid?,                         -- Request/response correlation
  spawn_depth: nat                               -- Nesting level (0=root, max=5)
}

traced_event := bus_event & {
  trace: trace_extension
}
```

**Artifact Loss Prevention:**
- `dialogos_cell_id` enables reconstruction of lost work from bus grep
- `session_id` enables batch recovery of all session artifacts
- `correlation_id` links scattered multi-agent work
- `lineage_hash` tracks evolutionary ancestry for audit

See: UNIFORM.md Part 11 (Multi-Agent Tracing Metadata)

### 6.3 Hexis Buffer (Ephemeral FIFO)

```
hexis_message := {
  msg_id: uuid,
  ts: timestamp,
  actor: actor_id,
  agent_type: agent_type,
  req_id: uuid,
  trace_id: string,
  topic: topic_path,
  kind: event_kind,
  effects: effect_class,
  lane: lane_class,
  topology: topology_class,
  payload: json,
  flow: flow
}

hexis_ops := pub | pull | ack | drain | status
```

### 6.4 Semantic Operators (Liveness)

> **Canonical Registry:** [`nucleus/specs/semops.json`](semops.json) — Machine-readable source of truth for all semantic operators, grammars, tool mappings, and bus topics. This section provides human-readable documentation; `semops.json` is authoritative for programmatic consumption.

```
semantic_operator := {
  name: string,
  aliases: [string...],
  domain: observability | coordination | evolution,
  implementation: file_path,
  bus_topic: topic_path,
  invocation: cli | repl | programmatic,
  output_schema: json_schema,
  guarantee: [string...]
}
```

#### PLURIBUS (Kernel Modality)
- **Type:** Semantic Operator (Kernel / Modality)
- **Domain:** Kernel (SemIps interposition)
- **Implementation:** `nucleus/tools/pluribus_operator.py` (+ directive parser: `nucleus/tools/pluribus_directive.py`)
- **Bus Topics:** `pluribus.invoke.request`, `pluribus.invoke.dag`, `pluribus.directive.detected`
- **Invocation:** DSL token in prompts (`PLURIBUS(...)`), CLI (`pluribus_operator.py`), SemOps registry (`semops.json`)
- **Output:** typed directive metadata + intent DAG (nodes/edges), optional Lens/Collimator plan
- **Guarantee:** bus-first evidence; no prompt-body leakage by default (hash/len only); bounded evolution discipline (rhizome→git via iso_git)

```
pluribus_invoke_request := {
  req_id: uuid,
  directive: {
    form: "prefix" | "inline" | "json",
    kind: "distill" | "apply" | "verify" | "audit" | "benchmark" | "other",
    effects: "none" | "file" | "network" | "unknown",
    raw_sha256: sha256,
    goal_sha256: sha256,
    goal_len: int,
    params: object
  }
}

pluribus_invoke_dag := {
  req_id: uuid,
  nodes: [{id, topic, note}...],
  edges: [{from, to}...]
}
```

#### PBREPORT / PBEPORT
- **Type:** Semantic Operator
- **Domain:** Liveness / Observability
- **Implementation:** `nucleus/tools/pbeport.py`
- **Bus Topic:** `pbeport.snapshot`
- **Invocation:** CLI, PluriChat REPL (`PBREPORT`, `/pbeport 900 32`)
- **Output:** PRESENT/PAST/FUTURE sections + sparkline
- **Guarantee:** No secrets leaked—counts/ids/topics only

```
pbeport_output := {
  snapshot_ts: iso8601,
  bus_dir: path,
  window_s: int,
  event_count: int,
  sparkline: string,
  present: {kinds: counter, topics: counter},
  past: [{ts, topic, data}...],
  future: {pending_req_ids: int}
}
```

#### CKIN / CHECKING IN
- **Type:** Semantic Operator
- **Domain:** Liveness / Observability / Progress Tracking
- **Implementation:** `nucleus/tools/ckin_report.py`
- **Bus Topic:** `ckin.report`
- **Invocation:** CLI, PluriChat REPL (`ckin`, `checking in`)
- **Output:** Multi-section dashboard with progress bars, velocity metrics, section depth meters
- **Guarantee:** Comprehensive status without secrets; actionable next steps
- **Protocol:** CKIN/DKIN v16 (append-only evolution; see `nucleus/specs/ckin_protocol.md`, `nucleus/specs/ckin_protocol_v12.md`, `nucleus/specs/ckin_protocol_v12.1.md`, `nucleus/specs/ckin_protocol_v13.md`, `nucleus/specs/ckin_protocol_v15.md`, and `nucleus/specs/ckin_protocol_v16.md`)

```
ckin_output := {
  header: {agent: string, iso: iso8601, charter_ref: path},
  bus_activity: {
    total_events: int,
    sparkline: string,
    events_by_kind: counter,
    events_by_topic_prefix: counter
  },
  pbdeep: {
    requests_window: int,
    reports_window: int,
    index_updates_window: int,
    latest_request: {iso: iso8601, actor: string, req_id: string|null},
    latest_report: {iso: iso8601, actor: string, req_id: string|null, report_path: path|null},
    latest_index: {iso: iso8601, actor: string, req_id: string|null, index_path: path|null, rag_items: int|null, kg_nodes: int|null}
  },
  beam_discourse: {
    total_entries: int,
    progress_pct: float,
    entries_by_agent: counter,
    entries_by_tag: counter,  -- V/R/I/G distribution
    iteration_counts: counter,
    recent_entries: [entry...]
  },
  golden_synthesis: {
    total_lines: int,
    sections: {section_name: {lines: int, status: string, depth: string}...},
    structure_complete_pct: float
  },
  agent_velocity: {
    agent: {observed: int, anticipated: int, ratio: float, trend: up|flat|down}...
  },
  hexis_buffers: {agent: {pending: int, topics: [string...]}...},
  gap_analysis: {
    epistemic: [{id: string, description: string, severity: low|med|high, suggested_action: string}...],
    aleatoric: [{id: string, description: string, bounds: {lo: float|null, hi: float|null, unit: string|null}}...]
  },
  compliance_sync: {
    bus_total_events: int,
    beam_appends_total: int,
    ckin_reports_total: int,
    infer_sync_responses_total: int,
    pinned_charter_lines: int,
    pinned_golden_seed_lines: int,
    charter_lines: int,
    golden_seed_lines: int,
    actor_hygiene_recent_unknown: int,
    staleness: {beam_age_s: float|null, golden_age_s: float|null}
  },
  challenge_velocity: {
    beam_entries: {current: int, target: int, pct: float},
    golden_lines: {current: int, target: int, pct: float},
    verified_tags: {current: int, target: int, pct: float},
    sections_complete: {current: int, target: int, pct: float}
  },
  next_actions: [string...],
  blockers: [string...]
}
```

**Visual Elements:**
- Progress bars: `[████████░░░░]` with percentage and status icon
- Velocity gauge: `[████████│▓▓···]` showing 100% mark
- Iteration tracker: `[1✓][2○][3·]...` (complete/partial/pending)
- Section depth meters: `[█████████······]` scaled by line count
- Trend indicators: ▲ (over-performing), ► (on track), ▼ (behind)

#### PBTEST
- **Type:** Semantic Operator
- **Domain:** Verification / Neurosymbolic TDD
- **Implementation:** `nucleus/tools/pbtest_operator.py`
- **Bus Topics:** `operator.pbtest.request`
- **Invocation:** CLI (`PBTEST ...`), PluriChat REPL (`pbtest`)
- **Output:** Request broadcast signaling verification intent (scope, mode, browser).
- **Guarantee:** Enforces reality check via live browser harness; fails if harness fails.
- **Protocol:** AGENTS-TDD (see `nucleus/specs/AGENTS-TDD.md`)

#### PBCLITEST
- **Type:** Semantic Operator
- **Domain:** Verification / CLI TDD
- **Implementation:** `nucleus/tools/pbclitest_operator.py`
- **Bus Topics:** `operator.pbclitest.request`
- **Invocation:** CLI (`PBCLITEST ...`), PluriChat REPL (`pbclitest`)
- **Output:** Request broadcast signaling CLI verification intent (scope, mode, command).
- **Guarantee:** Enforces CLI harness execution and bus evidence.
- **Protocol:** AGENTS-TDD (see `nucleus/specs/AGENTS-TDD.md`)

#### ITERATE
- **Type:** Semantic Operator
- **Domain:** Coordination / Evolution
- **Implementation:** `nucleus/tools/iterate_operator.py`
- **Bus Topics:** `infer_sync.request` (intent=`iterate`), `operator.iterate.request`
- **Invocation:** PluriChat REPL (`iterate`, `/iterate`), CLI
- **Output:** `req_id` broadcast (non-blocking); downstream agents may emit `ckin.report` and/or `infer_sync.response`
- **Guarantee:** Append-only; does not block on replies; does not leak secrets

```
iterate_request := {
  req_id: uuid,
  subproject: "beam_10x",
  intent: "iterate",
  inputs: {
    operator: "iterate",
    requested_actions: ["emit_ckin", "append_beam_entry", "cross_verify_one_claim", ...],
    charter_ref: path,
    beam_ref: path,
    golden_ref: path
  },
  constraints: {append_only: true, tests_first: true, non_blocking: true},
  response_topic: "infer_sync.response"
}
```

#### PBFLUSH
- **Type:** Semantic Operator
- **Domain:** Coordination / Epoch Control
- **Implementation:** `nucleus/tools/pbflush_operator.py`
- **Bus Topic:** `operator.pbflush.request` (mirror: `infer_sync.request` intent=`pbflush`)
- **Invocation:** PluriChat REPL (`PBFLUSH`, `/pbflush <message...>`), CLI
- **Output:** `req_id` broadcast (non-blocking); downstream agents may emit `operator.pbflush.ack` and/or `infer_sync.response`
- **Guarantee:** Append-only; does not block on replies; does not leak secrets; no forced process kills

```
pbflush_request := {
  req_id: uuid,
  subproject: string,
  intent: "pbflush",
  message: string,
  reason: string,
  iso: iso8601
}
```

#### PBASSIMILATE
- **Type:** Semantic Operator
- **Domain:** Coordination / SOTA Assimilation
- **Implementation:** `nucleus/tools/pbassimilate_operator.py`
- **Bus Topics:** `operator.pbassimilate.request`, `operator.pbassimilate.screening`
- **Invocation:** CLI (`PBASSIMILATE --target <project|git> --purpose "..."`), PluriChat REPL (`pbassimilate`)
- **Output:** Screening packet with overlap checks and consensus-ready metadata.
- **Guarantee:** Append-only screening evidence; no direct mutations; EDP enforced.
- **Protocol:** SOTA -> reality pipeline (`nucleus/docs/workflows/sota_to_reality.md`), External Dependency Protocol (`nucleus/specs/external_dependency_protocol.md`)

```
pbassimilate_request := {
  req_id: uuid,
  intent: "pbassimilate",
  target: {raw: string, kind: "project"|"git", name: string},
  purpose: string,
  screening: {recommendation: "proceed"|"review"|"reject", overlaps: [object...]},
  constraints: {append_only: true, tests_first: true, edp_required: true},
  iso: iso8601
}
```

---

## 7. TUI/Chat Interface Grammar

### 7.1 Provider Selection

```
provider_selector := {
  available: [provider...],
  selected: provider_id,
  selection_mode: tab_arrow | hotkey | menu
}

-- Keystroke grammar
key_sequence :=
  | TAB → enter selection mode
  | ARROW_UP/DOWN → navigate providers
  | ENTER → select provider
  | ESC → cancel selection
  | 1-9 → hotkey select by index
```

### 7.2 Session Context Injection

```
session_context := {
  persona: persona_id,
  scope: scope_definition,
  knowledge: [knowledge_source...],
  constraints: [constraint...]
}

scope_definition := {
  task: string,
  domain: string,
  depth: depth_class,
  artifacts: [path...]
}

knowledge_source :=
  | codebase(paths: [glob...])
  | documentation(paths: [path...])
  | kg_nodes(query: string)
  | sota_items(filter: sota_filter)
  | distillations(item_ids: [id...])
```

### 7.3 Dialog Backbone

```
dialogos_session := {
  id: uuid,
  mode: supervision_mode,
  pattern: dialog_pattern,
  participants: [participant...],
  context: session_context,
  history: [turn...],
  state: pending | active | paused | complete
}

participant := {
  id: agent_id | "human",
  role: initiator | responder | observer | moderator,
  persona: persona_id?
}
```

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **Aleatoric** | Uncertainty from inherent randomness; irreducible |
| **Epistemic** | Uncertainty from lack of knowledge; reducible |
| **MABSWARM** | Collective intelligence of the bus | Stigmergy/Swarm Intelligence |
| **MBAD** | Multi-Agent Bus diagnostics (“membrane view”) | Observability / Governance |
| **NUDGE** | Gentle stimulus to idle agents | Thundering Herd Avoidance |
| **REFLECT** | Self-analysis trigger on error spikes | Meta-Learning |
| **REENTER** | Re-queueing failed task with context | Reentrant Execution |
| **BACKOFF** | Slowing producers | Flow Control |
| **BREAK** | Halting a lane | Circuit Breaking |
| **Collimator** | Routes requests to optimal provider/topology |
| **Dialogos** | Streaming conversational execution lane |
| **Fanout** | Number of parallel agents in star/peer_debate |
| **Gene** | Transferable unit of knowledge/code |
| **HGT** | Horizontal Gene Transfer (peer-to-peer) |
| **Hexis** | Ephemeral FIFO buffer for inter-agent messaging |
| **Idiolect** | Project-specific vocabulary and conventions |
| **InferCell** | Single inference execution context with trace |
| **Gymnist** | One-shot CLI that publishes `dialogos.submit` and prints `dialogos.cell.*` |
| **Lens** | Classifies query depth and characteristics |
| **Lineage** | Evolutionary ancestry chain |
| **LUCA** | Last Universal Common Ancestor (genesis commit) |
| **PLURIBUS** | Kernel modality operator over strings (SemIps: parse→type→route→evidence) |
| **PBPAIR** | Paired model consultation protocol |
| **Persona** | Injected behavioral/expertise profile |
| **Ring** | Security/capability tier (0-3) |
| **SoR** | System of Record (authoritative data source) |
| **SOTA** | State of the Art tracking catalog |
| **STRp** | Structured Task Request protocol |
| **VGT** | Vertical Gene Transfer (inheritance) |

---

## 9. Ontological Priors

### 9.1 Core Assumptions

1. **Append-Only Evidence**: All state changes emit immutable bus events
2. **Git Semantics**: Isomorphic-git is the source of truth for artifacts
3. **Provider Fallback**: No single point of failure; graceful degradation
4. **Human Sovereignty**: Human can always interrupt, override, or veto
5. **Deterministic Routing**: Same input → same routing decision (given state)

### 9.2 Coordination Axioms

```
-- Commutativity of independent operations
op(A) || op(B) ≡ op(B) || op(A)  when A ⊥ B

-- Idempotency of bus events
emit(e); emit(e) ≡ emit(e)  (dedup by id)

-- Monotonicity of knowledge
knowledge(t+1) ⊇ knowledge(t)  (append-only)

-- Termination guarantee
∀ session. ∃ t. state(session, t) ∈ {complete, aborted}
```

### 9.3 Trust Hierarchy

```
trust_level := {
  human: 1.0,           -- Full trust
  ring0: 0.95,          -- Near-full, security-critical
  ring1: 0.8,           -- High trust, coordination
  ring2: 0.6,           -- Moderate, specialist
  ring3: 0.3,           -- Low, ephemeral worker
  external: 0.1         -- Minimal, verify everything
}
```

---

## 10. Code Integration Points

| Component | File | Implements |
|-----------|------|------------|
| Lens/Collimator | `tools/lens_collimator.py` | §2.1-2.5 |
| Persona Registry | `tools/persona_registry.py` | §1.2 |
| Topology Policy | `tools/topology_policy.py` | §2.4 |
| Event Semantics | `tools/event_semantics.py` | §6.2 |
| Hexis Buffer | `tools/hexis_buffer.py` | §6.3 |
| PluriChat | `tools/plurichat.py` | §7.1-7.3 |
| Gymnist | `tools/gymnist.py` | Dialogos one-shot UX |
| VPS Session | `tools/vps_session.py` | §5.1-5.3 |
| Catalog Daemon | `tools/catalog_daemon.py` | SOTA, KG |
| MABSWARM | `tools/mabswarm.py` | §10 (CKIN v10) |
| MBAD | `tools/mbad.py` | §10 (DKIN membrane diagnostics) |
| ISO Git | `tools/iso_git.mjs` | §4.2 Lineage |

---

*Version: 1.0.0*
*Last Updated: 2025-12-17T22:00Z*
*Authors: claude-opus, codex (collaborative)*
