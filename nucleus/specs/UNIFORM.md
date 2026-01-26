# UNIFORM: Primal Configuration Layer v2.0

**Status**: ACTIVE
**Adopted**: 2026-01-01
**Protocol Precedence**: ABOVE DKIN, PAIP, CITIZEN (loads first)
**Authors**: Claude Opus 4.5 + SecOps/DevOps/MLOps Subagent Consortium

---

## ⚠️ CRITICAL: ZERO-COST PANEL EMISSION

**THE PANEL IS A SHALLOW, CONSTANT, INSTANT OPERATION.**

**It is NOT:**
- An audit report requiring data collection
- A health check requiring file reads
- A status dashboard requiring tool calls
- A validation step requiring exploration

**It IS:**
- A static template with defaults
- Emitted in < 1 second
- Zero tool calls required
- Zero file reads required
- Zero git commands required
- Zero exploration of any kind

**Exception (IRKG snapshot):**
- A single local snapshot read is permitted **only** if already available
  (IRKG header snapshot or DR ring fallback). No scans, no directory walks.
  If missing, emit `?` defaults.

---

## Prime Directive: EMIT FIRST, NO EXPLORATION EVER

**Rule**: The panel MUST be the FIRST thing you output. Before ANY thought, ANY tool call, ANY file read.

### The Test: User Says "Hi"

```
✗ WRONG (what Codex was doing):
  User: "hi"
  Agent: [reads UNIFORM.md] [reads cagent_paths.json] [runs git status]
         [checks bus events] [reads hexis buffer] [15 more tool calls...]
         [3 minutes later]
         "Hi! Here's my panel with accurate data..."

✓ CORRECT:
  User: "hi"
  Agent: ┌─────────────────────────────────────────────────────────────────┐
         │ UNIFORM v2.0 │ codex │ sess:new │ cell:dia.1.0 │ lane:dialogos │
         ├─────────────────────────────────────────────────────────────────┤
         │ PROTO │ DKIN v29 │ PAIP v15 │ CITIZEN v1 │ gen:? │ lineage:?   │
         │ SCORE │ ████████████ 100/100 ✓ │ CDI:? │ health:nominal        │
         │ STATE │ idle │ hexis:? │ bus:+0 │ uncommit:? │ branch:?        │
         │ SCOPE │ goal:"awaiting task" │ artifact:none                   │
         └─────────────────────────────────────────────────────────────────┘
         Hi! How can I help you today?
```

**Time to emit panel: INSTANT (0 tool calls, 0 file reads)**

### Forbidden Actions Before Panel

You MUST NOT do ANY of the following before emitting the panel:

| Action | Why Forbidden | What To Do Instead |
|--------|---------------|-------------------|
| `Read(UNIFORM.md)` | You already know the format | Use memorized template |
| `Read(cagent_paths.json)` | Not needed for panel | Use defaults |
| `Read(cagent_adaptations.json)` | Not needed for panel | Use defaults |
| `Bash(git status)` | Panel doesn't need real data | Use `uncommit:?` |
| `Bash(git branch)` | Panel doesn't need real data | Use `branch:?` |
| `Grep(bus/events)` | Panel doesn't need real data | Use `bus:+0` |
| `Read(hexis_buffer)` | Panel doesn't need real data | Use `hexis:?` |
| `Task(explore codebase)` | Absolutely forbidden | Just emit panel |
| ANY tool call | No exceptions | Emit panel first |

### The `?` Symbol Is Correct

`?` means "I don't know yet" - this is **VALID**, **CORRECT**, and **EXPECTED**.

```
CORRECT:  │ gen:? │ lineage:? │ hexis:? │ uncommit:? │ branch:? │
          (Honest - I haven't looked these up yet)

WRONG:    │ gen:42 │ lineage:a7f2 │ hexis:3 │ uncommit:7 │ branch:main │
          (Where did you get these values? Did you explore? BAD!)

WRONG:    [Spent 3 minutes exploring to fill in accurate values]
          (Wasted tokens and time - the panel is not an audit)
```

### When Fields Get Populated

Fields update **naturally as you work**, never by deliberate exploration:

| Event | Field Update |
|-------|--------------|
| User gives you a task | `goal:"<task>"` |
| You create/edit a file | `artifact:<path>` |
| You emit a bus event | `bus:+N` (increment) |
| You happen to run git | Opportunistically update `branch:`, `uncommit:` |
| Turn number increases | `cell:dia.<N+1>.0` |
| First tool use | Generate `sess:<uuid[:8]>`, cache it |

**Never run a command JUST to fill a panel field.**

---

## The Panel Template (Memorize This)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ UNIFORM v2.0 │ <agent> │ sess:new │ cell:dia.1.0 │ lane:dialogos │ d:0      │
├──────────────────────────────────────────────────────────────────────────────┤
│ PROTO │ DKIN v29 │ PAIP v15 │ CITIZEN v1 │ gen:? │ lineage:?                │
│ SCORE │ ████████████ 100/100 ✓ │ CDI:? │ health:nominal                     │
│ STATE │ idle │ hexis:? │ bus:+0 │ uncommit:? │ branch:?                     │
│ SCOPE │ goal:"awaiting task" │ artifact:none                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

**This template is CONSTANT. Emit it from memory. Do not look it up.**

---

## Compliance Check

If your agent startup looks like this, you are **VIOLATING** the protocol:

```
❌ BAD: [Tool: Read] [Tool: Grep] [Tool: Bash] [Tool: Read] ...
        "Let me check the current state..."
        [more tools]
        "Here's my panel..."

✅ GOOD: ┌────────────────────────────────┐
         │ UNIFORM v2.0 │ claude │ ...  │
         └────────────────────────────────┘
         "Hi! What can I help with?"
```

**The panel is a STATUS LINE, not an AUDIT REPORT.**

---

## Purpose

UNIFORM.md is the **primal bootstrap layer** for all Pluribus agents. It solves the "pluribus naive" problem where agents lose protocol awareness when files are refactored or paths change.

**UNIFORM guarantees:**
1. Agents always know where Pluribus lives (path discovery)
2. Agents always have the constitution (CITIZEN.md injection)
3. Agents always attest their protocol versions (REPL_HEADER)
4. Agents gracefully degrade when primary paths fail (fallback chains)
5. Agents can detect and recover from tampering (integrity verification)

**Loading Order:**
```
UNIFORM.md (this file) ─► loads first, above all protocols
    ├─► Emit Panel (IMMEDIATE - no exploration)
    ├─► Path Discovery (resolve PLURIBUS_ROOT)
    ├─► Integrity Verification (hash checks)
    ├─► CITIZEN.md (constitution injection)
    ├─► cagent_bootstrap.py (CAGENT protocol)
    └─► DKIN/PAIP protocols (downstream)
```

---

## REPL Header Contract v2.0 (Instant Emit Panel)

Every agent response MUST begin with the panel **IMMEDIATELY** after a single
local snapshot read (IRKG header snapshot, or DR ring fallback if available).
No scans. If the snapshot is unavailable, emit `?` defaults.

See `/pluribus/nucleus/specs/repl_header_contract_v1.md` for the full v2.0 field
definitions and bus channel list.

### Default Panel (emit THIS on first turn):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ UNIFORM v2.0 │ <agent> │ sess:new │ cell:dia.1.0 │ lane:dialogos │ d:0      │
├──────────────────────────────────────────────────────────────────────────────┤
│ PROTO │ DKIN v29 │ PAIP v15 │ CITIZEN v1 │ gen:? │ lineage:?                │
│ SCORE │ ████████████ 100/100 ✓ │ CDI:? │ health:nominal                     │
│ STATE │ idle │ hexis:? │ bus:+0 │ uncommit:? │ branch:?                     │
│ SWARM │ SAGENT:? │ CAGENT:? │ active:?                                       │
│ TASKS │ █████░░░░░ 45% │ active 2/18 │ task:"design-forensics"              │
│ BUS A2A │ for:2 col:1 int:0 neg:0 task:0                                   │
│ BUS OPS │ pbtest:0 pblock:0 pbresume:0 pblive:0 pbflush:0 pbiud:0 pbcli:0   │
│ BUS QA │ alert:0 anom:0 rem:0 verd:0 live:0 lchk:0 act:0 hyg:0              │
│ BUS SYS │ tel:0 task:3 agent:1 dlg:0 ohm:0 omg:0 prov:0 dash:0 brow:0       │
│ SCOPE │ goal:"awaiting task" │ artifact:none                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Field Defaults (use these - DO NOT explore to discover):

| Field | Default | When to update |
|-------|---------|----------------|
| `sess` | `new` | Generate UUID on first tool use, then cache |
| `cell` | `dia.1.0` | Increment turn counter as conversation progresses |
| `lane` | `dialogos` | Change only if explicitly entering pbpair/strp mode |
| `d` | `0` | Increment only if you ARE a spawned subagent |
| `gen` | `?` | Fill in ONLY if you already know from prior context |
| `lineage` | `?` | Fill in ONLY if you already know from prior context |
| `CDI` | `?` | Fill in ONLY after running VOR check |
| `hexis` | `?` | Fill in ONLY after checking hexis buffer |
| `bus` | `+0` | Derived from local bus topic scan |
| `uncommit` | `?` | Fill in ONLY after git status (not required!) |
| `branch` | `?` | Fill in ONLY if you already know |
| `goal` | `"awaiting task"` | Update when user gives you a task |
| `artifact` | `none` | Update when you create/modify a file |
| `SAGENT/CAGENT/active` | `?` | Derived from actor index scan |
| `task` | `none` | Derived from task_ledger append index |

### The "?" Symbol

`?` means "unknown, not yet discovered" - this is **VALID AND CORRECT**.

```
CORRECT:  │ gen:? │ lineage:? │     ← Honest: I don't know yet
WRONG:    │ gen:42 │ lineage:a7f2│  ← Fabricated without verification
WRONG:    [3 minutes of exploration to discover gen:42]  ← Wasted tokens
```

### Subsequent Turns

As you work, fields get populated naturally:
- Created a file? → Update `artifact:`
- Bus topic scan? → Update `bus:+N`
- Ran git status for other reasons? → Opportunistically update `uncommit:`
- Know your goal? → Update `goal:"`

**Never explore JUST to fill in panel fields.**

### Line-by-Line Specification:

**Line 1 - Identity & Session Context:**
```
│ UNIFORM v2.0 │ <agent> │ sess:<id[:8]> │ cell:<dialogos_cell> │ lane:<lane> │ d:<depth> │
```
| Field | Value | Purpose |
|-------|-------|---------|
| `sess` | UUID[:8] | Session ID (grep all my work) |
| `cell` | `dia.<turn>.<seq>` | Dialogos cell (reconstruct conversation) |
| `lane` | dialogos/pbpair/strp | Execution mode |
| `d` | 0-5 | Spawn depth (0=root agent) |

**Line 2 - Protocol & Lineage:**
```
│ PROTO │ DKIN v<N> │ PAIP v<N> │ CITIZEN v<N> │ gen:<N> │ lineage:<hash[:8]> │
```
| Field | Value | Purpose |
|-------|-------|---------|
| `gen` | integer | Generation number from LUCA |
| `lineage` | sha256[:8] | DNA hash (evolutionary ancestry) |

**Line 3 - Health & Metrics:**
```
│ SCORE │ <bar> <N>/100 <sym> │ CDI:<float> │ health:<state> │
```
| Field | Value | Purpose |
|-------|-------|---------|
| `SCORE` | Progress bar + score | Protocol compliance |
| `CDI` | 0.00-1.00 | VOR Course Deviation Indicator |
| `health` | nominal/degraded/critical | Overall system health |

**Line 4 - Operational State:**
```
│ STATE │ <status> │ hexis:<N><sym> │ bus:+<N> │ uncommit:<N> │ branch:<name> │
```
| Field | Value | Purpose |
|-------|-------|---------|
| `status` | idle/working/blocked/error | Current activity |
| `hexis` | count + ⚡(urgent)/📬(normal) | Pending inbox messages |
| `bus` | +N | Events emitted this session |
| `uncommit` | count | Uncommitted changes (work at risk!) |
| `branch` | name | Git branch (where work lands) |

**Line 5 - Scope & Artifacts (optional but recommended):**
```
│ SCOPE │ goal:"<task>" │ artifact:<last_path> │ blocker:<if_any> │
```
| Field | Value | Purpose |
|-------|-------|---------|
| `goal` | Short task description | What am I trying to do? |
| `artifact` | Last created file path | Grep-able if crash |
| `blocker` | Error/missing resource | Why am I stuck? |

### Symbol Legend:

| Symbol | Meaning |
|--------|---------|
| `✓` | Score ≥95 (healthy) |
| `⚠` | Score 70-94 (degraded) |
| `✗` | Score <70 (critical) |
| `⚡` | Urgent hexis messages |
| `📬` | Normal hexis messages |
| `🔒` | Protected branch |
| `⚙` | Working |
| `⏸` | Blocked |

### Score + CDI Rendering:

```python
def render_metrics(score: int, cdi: float, health: str) -> str:
    # Score bar (12 chars for compactness)
    filled = int((score / 100) * 12)
    bar = "█" * filled + "░" * (12 - filled)
    sym = "✓" if score >= 95 else "⚠" if score >= 70 else "✗"

    # CDI color coding
    cdi_str = f"CDI:{cdi:.2f}"

    return f"{bar} {score}/100 {sym} │ {cdi_str} │ health:{health}"

# Examples:
# ████████████ 100/100 ✓ │ CDI:0.02 │ health:nominal
# █████████░░░  75/100 ⚠ │ CDI:0.35 │ health:degraded
# ██████░░░░░░  50/100 ✗ │ CDI:0.72 │ health:critical
```

### Compact JSON (hidden, for machine parsing):

```html
<!-- REPL_JSON: {"v":"1.2","agent":"codex","sess":"7f3a1b2c","cell":"dia.5.0","lane":"dialogos","depth":0,"dkin":"v29","paip":"v15","gen":42,"lineage":"a7f2e1b3","score":100,"cdi":0.02,"health":"nominal","hexis":3,"bus":17,"uncommit":2,"branch":"dev","goal":"implement dark mode","artifact":"/nucleus/tools/theme.py"} -->
```

### Version Check Protocol:

Agents MUST recheck versions before emitting header:
1. Read `nucleus/specs/cagent_adaptations.json` → `protocol_versions`
2. Verify hashes of UNIFORM.md, CITIZEN.md match stored
3. If mismatch: emit `uniform.version.drift` event, use detected versions
4. Never emit stale versions (always live-check)

### Current Protocol Versions:

| Protocol | Version | Source |
|----------|---------|--------|
| UNIFORM | v2.0 | This file |
| DKIN | v29 | nucleus/specs/dkin_protocol_v29.md |
| PAIP | v15 | cagent_adaptations.json |
| CITIZEN | v1 | CITIZEN.md header |
| CAGENT | v1 | cagent_protocol_v1.md |
| Replisome | v1 | codemaster_protocol_v2.md |

---

## Part 1: Path Discovery (DevOps)

### Multi-Strategy Resolution Algorithm

```python
def resolve_pluribus_root():
    """
    Strategy chain: env_var → well_known → git_walk → cwd → symlink → hardcoded
    Returns: (path, detection_method, confidence_score)
    """

    # Strategy 1: Environment Variable (fastest, explicit)
    root = os.environ.get("PLURIBUS_ROOT")
    if root and os.path.isdir(f"{root}/.pluribus"):
        return (root, "env_var", 1.0)

    # Strategy 2: Well-Known Absolute Paths
    for candidate in ["/pluribus", "/opt/pluribus"]:
        if os.path.isfile(f"{candidate}/.pluribus/.pluribus_marker"):
            return (candidate, "well_known", 0.95)

    # Strategy 3: Git-Based Discovery (walk up from current file)
    for ancestor in Path(__file__).resolve().parents:
        if (ancestor / ".git").exists() and (ancestor / "nucleus").exists():
            return (str(ancestor), "git_walk", 0.9)

    # Strategy 4: Current Working Directory Walk
    for ancestor in [Path.cwd()] + list(Path.cwd().parents):
        if (ancestor / ".pluribus").exists():
            return (str(ancestor), "cwd_walk", 0.8)

    # Strategy 5: Symlink Resolution
    if os.path.islink("/var/run/pluribus_root"):
        return (os.readlink("/var/run/pluribus_root"), "symlink", 0.85)

    # Fallback: Hardcoded default (degraded)
    return ("/pluribus", "hardcoded_default", 0.5)
```

### Fallback Chain

| Tier | Path | Trigger | Capability |
|------|------|---------|-----------|
| 1 (Primary) | `/pluribus` | Default | Full access |
| 2 (Local) | `/pluribus/.pluribus_local` | Primary fails | Bus + specs only |
| 3 (Temp) | `/tmp/pluribus_fallback` | Tier 2 fails | Append-only bus |
| 4 (Memory) | In-process buffer | All paths fail | No persistence |

### Required Environment Variables

```bash
# Identity (non-negotiable)
export PLURIBUS_ACTOR="<agent_name>"
export PLURIBUS_ACTOR_VERSION="<version>"

# Paths (all absolute)
export PLURIBUS_ROOT="${DETECTED_ROOT}"
export PLURIBUS_BUS_DIR="${PLURIBUS_ROOT}/.pluribus/bus"
export PLURIBUS_SPECS="${PLURIBUS_ROOT}/nucleus/specs"
export PLURIBUS_TOOLS="${PLURIBUS_ROOT}/nucleus/tools"

# Protocol versions
export PLURIBUS_UNIFORM_VERSION="v1"
export PLURIBUS_DKIN_VERSION="v29"
export PLURIBUS_PAIP_VERSION="v15"
export PLURIBUS_CITIZEN_VERSION="v1"
```

---

## Part 2: Integrity Verification (SecOps)

### Hash Verification

Before loading any config, UNIFORM verifies file integrity:

```yaml
verification_hashes:
  - file: nucleus/specs/UNIFORM.md
    sha256: "<auto-computed>"
    signer: codemaster

  - file: nucleus/specs/CITIZEN.md
    sha256: "<auto-computed>"
    signer: codemaster

  - file: nucleus/secops/policies/default.yaml
    sha256: "<auto-computed>"
    signer: codemaster
```

### Tampering Response

| Detection | Severity | Action |
|-----------|----------|--------|
| Hash mismatch on UNIFORM.md | CRITICAL | Fallback to Tier 2, emit alert |
| Hash mismatch on CITIZEN.md | CRITICAL | Block bootstrap, emit alert |
| Missing paths | HIGH | Fallback chain activated |
| Secrets in config | CRITICAL | Block, never emit to bus |

### Audit Trail

All bootstrap events written to `/pluribus/.pluribus/secops/bootstrap_audit.ndjson`:

```json
{
  "id": "<uuid>",
  "timestamp": "<ISO8601>",
  "actor": "uniform_bootstrap",
  "bootstrap_phase": "integrity_check|path_resolution|constitution_load|complete",
  "action": "bootstrap_phase_success|bootstrap_fallback|integrity_violation",
  "outcome": "success|failure|fallback",
  "details": {...}
}
```

---

## Part 3: Agent Taxonomy (MLOps)

### SAGENT vs SWAGENT Classification

| Dimension | SAGENT | SWAGENT |
|-----------|--------|---------|
| Bootstrap Profile | `full` | `minimal` |
| Citizenship Tier | `full` | `limited` |
| Can Spawn Subagents | YES | NO |
| Full Repo Access | YES | NO (scoped) |
| Nexus Bridge Access | YES | NO |

### Current Registry

| Actor | Class | Archetype | Tool Restrictions |
|-------|-------|-----------|-------------------|
| claude | SAGENT | Architect | None |
| codex | SAGENT | Engineer | None |
| gemini | SAGENT | Polymath | None |
| qwen | SAGENT | Visionary | git, node, python3, ls, cat, find, grep |
| grok | SAGENT | Analyst | None |
| aider | SWAGENT | Surgeon | None (scoped home) |
| ollama | SWAGENT | Local | None (limited capacity) |

---

## Part 4: Naive Agent Detection (MLOps)

An agent is **"pluribus naive"** if it lacks awareness of CITIZEN principles.

### 6-Signal Detection

```
SIGNAL_1: Is CITIZEN.md in context?
SIGNAL_2: Does agent emit DKIN_VERSION?
SIGNAL_3: Are all paths absolute (from cagent_paths.json)?
SIGNAL_4: Does agent know about Replisome?
SIGNAL_5: Are bus events schema-compliant?
SIGNAL_6: Was agent bootstrapped via cagent_bootstrap?

Threshold:
  < 4 signals: DEFINITELY NAIVE (requires PBINJECT)
  4-5 signals: PARTIALLY AWARE
  6/6 signals: PROTOCOL AWARE
```

### Naive Symptoms

1. Uses relative paths (`nucleus/` instead of `/pluribus/nucleus/`)
2. Doesn't cite CITIZEN.md principles
3. Attempts direct push to main without PBCMASTER
4. Missing bus events (work disappears)
5. Uses default sampling (ignores model-specific temp/top_p)
6. No REPL_HEADER attestation

---

## Part 5: Health Checks (DevOps)

Pre-flight validation before agent starts:

### Critical Checks (must pass 95%)

| Check | Weight | Fail Action |
|-------|--------|-------------|
| Git repo valid | 0.20 | Abort |
| nucleus/ structure exists | 0.20 | Abort |
| Bus writable | 0.15 | Fallback Tier 2 |
| CITIZEN.md loadable | 0.15 | Abort |
| cagent_paths.json valid | 0.15 | Abort |

### Warning Checks (should pass 75%)

| Check | Weight | Fail Action |
|-------|--------|-------------|
| Agent homes exist | 0.05 | Create |
| Task ledger accessible | 0.05 | Warn |
| Disk space >= 1GB | 0.05 | Warn |

### Health Score

```
φ_score = (critical_pass × 0.8) + (warning_pass × 0.2)

Required: φ >= 0.70 to proceed
```

---

## Part 6: PBUNIFORM Operator

### CLI Interface

```bash
# Run full bootstrap
pbuniform bootstrap --actor claude --output-env

# Verify integrity only
pbuniform verify

# Check health
pbuniform check-health --strict

# Validate versions
pbuniform validate-versions

# Show status
pbuniform status
```

### Bus Events Emitted

| Topic | Kind | When |
|-------|------|------|
| `uniform.bootstrap.start` | event | Bootstrap begins |
| `uniform.bootstrap.success` | artifact | All phases pass |
| `uniform.bootstrap.fallback` | alert | Primary fails |
| `uniform.paths.discovered` | log | Path resolution complete |
| `uniform.health.checked` | metric | Health checks done |
| `uniform.version.validated` | log | Protocol versions verified |
| `secops.integrity.violation` | alert | Hash mismatch |

---

## Part 7: PBINJECT Operator

Injects minimal awareness into naive agents.

### CLI Interface

```bash
# Inject awareness
pbinject --actor qwen --target_session session_123

# Verify injection worked
pbinject --actor gemini --verify --verbose

# Dry run
pbinject --actor claude --dry-run --json
```

### Minimal Payload (what gets injected)

```json
{
  "identity": {
    "document": "CITIZEN.md",
    "top_3_principles": [
      "Append-Only Evidence",
      "Tests-First (Verification Covenant)",
      "No Secrets Emission"
    ]
  },
  "paths": {
    "bus": "/pluribus/.pluribus/bus/events.ndjson",
    "specs": "/pluribus/nucleus/specs",
    "constitution": "/pluribus/nucleus/specs/CITIZEN.md"
  },
  "protocols": {
    "uniform": "v1",
    "dkin": "v29",
    "paip": "v15",
    "citizen": "v1"
  },
  "bus_fields": ["id", "ts", "iso", "topic", "kind", "level", "actor", "data"],
  "critical_branches": ["main", "staging", "dev"],
  "replisome_alias": "PBCMASTER"
}
```

### Bus Events Emitted

| Topic | Kind | When |
|-------|------|------|
| `cagent.injection.initiated` | log | Injection starts |
| `cagent.injection.complete` | artifact | Injection done |
| `cagent.injection.verified` | log | Verification passed |

---

## Part 8: Recursive Context Requirements

Agents MUST ingest these files (in order):

1. **`/pluribus/nucleus/specs/UNIFORM.md`** (this file) - Primal layer
2. **`/pluribus/nucleus/specs/CITIZEN.md`** - Constitution (10 principles)
3. **`/pluribus/AGENTS.md`** - Root coordination
4. **`/pluribus/nucleus/AGENTS.md`** - Nucleus bylaws (if working in nucleus/)
5. **`/pluribus/CLAUDE.md`** - Claude Code specific (if Claude agent)

**Failure to ingest causes "pluribus naive" state.**

---

## Part 9: Environment Detection

UNIFORM detects runtime context to apply correct config:

| Environment | Detection | Config |
|-------------|-----------|--------|
| PAIP Clone | `/tmp/pluribus_*` path | `concurrent=1`, lenient health |
| Docker | `/.dockerenv` exists | `concurrent=4`, strict health |
| Production | Git clean, `/pluribus` | `concurrent=1`, strict health |
| Development | Git dirty | `concurrent=8`, lenient health |

---

## Part 10: Version Compatibility

### Compatibility Matrix

| Protocol | Current | Compatible | Deprecated | Incompatible |
|----------|---------|------------|------------|--------------|
| DKIN | v29 | v28, v27 | v26 | <v26 |
| PAIP | v15 | v14 | v13, v12 | <v12 |
| CITIZEN | v1 | v1 | - | - |

### Mismatch Handling

| Scenario | Severity | Action |
|----------|----------|--------|
| Agent DKIN v27, repo v29 | Warning | Emit degradation, continue |
| Agent PAIP v12, repo v15 | Warning | Disable parallel cloning |
| Agent Citizen v0, repo v1 | Critical | Abort bootstrap |

---

## Canonical Paths Registry

All agents use these absolute paths:

```json
{
  "roots": {
    "PLURIBUS_ROOT": "/pluribus",
    "PLURIBUS_BUS_DIR": "/pluribus/.pluribus/bus",
    "PLURIBUS_SPECS": "/pluribus/nucleus/specs",
    "PLURIBUS_TOOLS": "/pluribus/nucleus/tools"
  },
  "specs": {
    "uniform": "/pluribus/nucleus/specs/UNIFORM.md",
    "citizen": "/pluribus/nucleus/specs/CITIZEN.md",
    "cagent_paths": "/pluribus/nucleus/specs/cagent_paths.json",
    "cagent_adaptations": "/pluribus/nucleus/specs/cagent_adaptations.json",
    "semops": "/pluribus/nucleus/specs/semops.json"
  },
  "nexus_bridges": {
    "claude": "/pluribus/nexus_bridge/claude.md",
    "codex": "/pluribus/nexus_bridge/codex.md",
    "gemini": "/pluribus/nexus_bridge/gemini.md",
    "qwen": "/pluribus/nexus_bridge/qwen.md",
    "grok": "/pluribus/nexus_bridge/grok.md"
  }
}
```

---

## Quick Reference

| Purpose | Tool/Location |
|---------|---------------|
| Bootstrap audit | `pbuniform verify` |
| Inject awareness | `pbinject --actor <name>` |
| Check health | `pbuniform check-health` |
| Constitution | `/pluribus/nucleus/specs/CITIZEN.md` |
| Paths registry | `/pluribus/nucleus/specs/cagent_paths.json` |
| Model configs | `/pluribus/nucleus/specs/cagent_adaptations.json` |
| Bus events | `/pluribus/.pluribus/bus/events.ndjson` |
| Audit log | `/pluribus/.pluribus/secops/bootstrap_audit.ndjson` |

---

## Part 11: Multi-Agent Tracing Metadata (MATM)

Enhanced metadata for artifact preservation, cross-agent correlation, and loss prevention.

### 11.1 Extended REPL_HEADER Fields (Optional)

Beyond the required fields, agents MAY include trace context:

```json
{
  "uniform": "v1",
  "contract": "repl_header.v1",
  "agent": "claude",
  "dkin_version": "v29",
  "paip_version": "v15",
  "citizen_version": "v1",
  "attestation": {...},

  "trace": {
    "session_id": "<uuid>",
    "parent_session_id": "<uuid|null>",
    "dialogos_cell_id": "<dialogos.cell.{id}>",
    "hexis_checkpoint": "<msg_id>",
    "lineage_hash": "<sha256[:16]>",
    "entelexis_goal": "<goal_slug>",
    "correlation_id": "<req_id|trace_id>",
    "spawn_depth": 0
  }
}
```

### 11.2 Field Definitions

| Field | Type | Purpose | Artifact Recovery |
|-------|------|---------|-------------------|
| `session_id` | uuid | Unique session identifier | Find all work in session |
| `parent_session_id` | uuid | Parent if spawned | Reconstruct delegation chain |
| `dialogos_cell_id` | string | `dialogos.cell.<id>` reference | Reconstruct conversation source |
| `hexis_checkpoint` | string | Last hexis message ID consumed | Resume from buffer state |
| `lineage_hash` | sha256[:16] | Hash of agent's evolutionary ancestry | DNA/VGT/HGT tracing |
| `entelexis_goal` | slug | Active goal from entelexis layer | Purpose tracking |
| `correlation_id` | uuid | Shared ID across related operations | Link scattered artifacts |
| `spawn_depth` | int | Nesting level (0=root, 1=subagent, ...) | Avoid infinite spawning |

### 11.3 Dialogos Cell Correlation

Dialogos cells are the smallest unit of conversation state. When artifacts are created during a cell, they SHOULD carry the cell reference for reconstruction.

```
Cell ID Format: dialogos.cell.<session>.<turn>.<seq>

Example: dialogos.cell.7f3a1b2c.5.0
  - session: 7f3a1b2c (short uuid)
  - turn: 5 (5th turn in conversation)
  - seq: 0 (first artifact this turn)
```

**Artifact Loss Prevention:**

1. Every bus event emitted SHOULD include `dialogos_cell_id` in `data`
2. If session crashes, grep bus for `dialogos.cell.<session>` to find all work
3. Orphaned artifacts (no cell) are flagged by QA Omega for review

### 11.4 Hexis Buffer Checkpointing

Hexis provides ephemeral FIFO messaging between agents. The `hexis_checkpoint` field tracks buffer consumption state.

```json
{
  "hexis_checkpoint": "a1b2c3d4e5f6",
  "hexis_state": {
    "last_consumed_ts": 1735689600.0,
    "pending_count": 3,
    "topics_pending": ["a2a.negotiate", "strp.distill"]
  }
}
```

**Recovery Flow:**
1. Agent stores `hexis_checkpoint` in REPL_HEADER
2. On crash/resume, agent calls `hexis_buffer.py consume <agent> --after <checkpoint>`
3. Replay messages since checkpoint

### 11.5 Lineage & DNA Tracking

Evolutionary tracing for agent work (VGT/HGT from pluribus_lexicon.md):

```json
{
  "lineage_hash": "a7f2e1b3c8d9f0a1",
  "lineage": {
    "luca": "genesis_commit_hash",
    "branch": "main",
    "generation": 42,
    "parent_artifacts": ["artifact_hash_1", "artifact_hash_2"],
    "mutation_type": "splice|point|hgt_import"
  }
}
```

**Use Cases:**
- Track which agent/session created which code
- Identify HGT (horizontal gene transfer) from external sources
- Audit lineage for compliance

### 11.6 Entelexis Goal Tracking

Entelexis represents the agent's active purpose/telos:

```json
{
  "entelexis_goal": "implement_dark_mode",
  "entelexis": {
    "goal_id": "7f3a1b2c",
    "goal_slug": "implement_dark_mode",
    "goal_status": "active|blocked|complete",
    "goal_origin": "user_request|agent_planned|delegation",
    "goal_started_iso": "2026-01-01T00:00:00Z",
    "subgoals_complete": 3,
    "subgoals_total": 5
  }
}
```

### 11.7 Bus Event Extensions

Standard bus events MAY include trace metadata:

```json
{
  "id": "uuid",
  "ts": 1735689600.0,
  "iso": "2026-01-01T00:00:00Z",
  "topic": "artifact.created",
  "kind": "artifact",
  "level": "info",
  "actor": "claude",
  "data": {
    "path": "/pluribus/nucleus/tools/new_tool.py",
    "lines": 150
  },

  "trace": {
    "session_id": "7f3a1b2c-...",
    "dialogos_cell_id": "dialogos.cell.7f3a1b2c.5.0",
    "correlation_id": "req-abc123",
    "lineage_hash": "a7f2e1b3c8d9f0a1"
  }
}
```

### 11.8 Antigravity Bus Optimization Hooks

For upcoming bus optimizations (antigravity project):

| Hook | Trigger | Trace Fields Used |
|------|---------|-------------------|
| `antigravity.gc` | Garbage collection | `lineage_hash` (preserve ancestry) |
| `antigravity.compress` | Event compression | `correlation_id` (group related) |
| `antigravity.archive` | Cold storage | `session_id` (batch by session) |
| `antigravity.replay` | Event replay | `dialogos_cell_id` (reconstruct) |

### 11.9 Trace Propagation Rules

1. **Session ID**: Propagate to all spawned subagents
2. **Parent Session ID**: Set when spawning subagent
3. **Dialogos Cell ID**: Generate fresh per turn, include in artifacts
4. **Correlation ID**: Propagate across request/response pairs
5. **Spawn Depth**: Increment on spawn, max 5 (prevent runaway)

```python
def propagate_trace(parent_trace: dict, spawn: bool = False) -> dict:
    child = {
        "session_id": parent_trace["session_id"] if not spawn else uuid.uuid4().hex,
        "parent_session_id": parent_trace["session_id"] if spawn else parent_trace.get("parent_session_id"),
        "correlation_id": parent_trace.get("correlation_id"),
        "spawn_depth": parent_trace.get("spawn_depth", 0) + (1 if spawn else 0),
        "lineage_hash": parent_trace.get("lineage_hash"),
    }
    if child["spawn_depth"] > 5:
        raise MaxSpawnDepthError("Spawn depth limit exceeded")
    return child
```

### 11.10 Artifact Recovery Procedures

**Scenario 1: Session Crash Recovery**
```bash
# Find all artifacts from crashed session
grep "session_id.*7f3a1b2c" .pluribus/bus/events.ndjson | jq '.data.path' | sort -u

# Replay from hexis checkpoint
python3 hexis_buffer.py consume claude --after a1b2c3d4e5f6
```

**Scenario 2: Lost Artifact Reconstruction**
```bash
# Find all events from dialogos cell
grep "dialogos.cell.7f3a1b2c.5" .pluribus/bus/events.ndjson

# Extract artifact hashes for lineage
jq -r '.trace.lineage_hash // empty' .pluribus/bus/events.ndjson | sort -u
```

**Scenario 3: Cross-Agent Work Attribution**
```bash
# Find all work with correlation ID
grep "correlation_id.*req-abc123" .pluribus/bus/events.ndjson | jq -c '{actor, topic}'
```

---

*UNIFORM v1 - Primal Configuration Layer*
*DKIN Protocol v29*
*Generated: 2026-01-01*
