# Isomorphic Git Reference

Complete reference for Pluribus's isomorphic-git integration, covering the pure-JS Git implementation, evolutionary branching system, HGT Guard Ladder, and HTTP API.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Pluribus Git Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Dashboard UI │───▶│ git_server.py│───▶│ iso_git.mjs          │  │
│  │ (GitView)    │    │ (HTTP :9300) │    │ (Pure-JS Git)        │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │               │
│         │                   ▼                      ▼               │
│         │            ┌─────────────┐        ┌─────────────┐       │
│         └───────────▶│ Pluribus Bus│◀───────│ .git/       │       │
│                      │ events.ndjson│       │ Repository  │       │
│                      └─────────────┘        └─────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `iso_git.mjs` | `nucleus/tools/iso_git.mjs` | Pure-JS Git CLI using isomorphic-git |
| `git_server.py` | `nucleus/tools/git_server.py` | HTTP API server (port 9300) |
| Dashboard GitView | `dashboard/src/components/GitView.tsx` | Visual DAG explorer |
| Bus Events | `$PLURIBUS_BUS_DIR/events.ndjson` | Audit trail for all Git operations |

---

## iso_git.mjs - Pure-JS Git CLI

### Core Commands

```bash
# Initialize repository
node nucleus/tools/iso_git.mjs init <directory>

# Commit all changes
node nucleus/tools/iso_git.mjs commit <directory> "commit message"

# View commit log
node nucleus/tools/iso_git.mjs log <directory>

# View commit details + diff
node nucleus/tools/iso_git.mjs show <directory> <commit-sha>

# Check working tree status
node nucleus/tools/iso_git.mjs status <directory>

# Branch operations
node nucleus/tools/iso_git.mjs branch <directory>              # list
node nucleus/tools/iso_git.mjs branch <directory> <name>       # create

# Checkout branch
node nucleus/tools/iso_git.mjs checkout <directory> <branch>
```

### Advanced Commit Log (JSON + DAG + Trailers)

`iso_git.mjs log` returns a JSON payload with commit metadata that is designed for graphing and provenance:

```json
{
  "commits": [
    {
      "sha": "83c865f...",
      "message": "fix(dashboard): gate e2e hydration hooks\n\nreq_id: ...\nartifact: ...",
      "author": "codex",
      "email": "codex@local",
      "date": "2025-12-19T07:02:01Z",
      "parents": ["972475b..."]
    }
  ]
}
```

Usage notes:

- **DAG reconstruction:** use `parents[]` to build the commit graph and to highlight merges (multiple parents).
- **Trailer parsing:** commit message trailers (e.g., `req_id`, `artifact`, `pqc`) sit after a blank line; parse with a simple `key: value` regex for provenance.
- **Diff inspection:** use `iso_git.mjs show <sha>` to get file-level diffs and blob IDs.
- **UI integration:** the dashboard GitView consumes this shape via `GET /git/log` (which wraps `iso_git.mjs log`).
- **Depth:** `iso_git.mjs log` returns the last 50 commits by default.

### Evolutionary Commands (`evo`)

The evolutionary system enables controlled code transfer between lineages:

```bash
# Create evolutionary branch (VGT - Vertical Gene Transfer)
node nucleus/tools/iso_git.mjs evo branch <directory> <name> [parent-dag-id]

# List evolutionary branches with lineage info
node nucleus/tools/iso_git.mjs evo list <directory>

# Horizontal Gene Transfer - transfer code between lineages
node nucleus/tools/iso_git.mjs evo hgt <target-dir> <source-ref>

# View lineage DAG
node nucleus/tools/iso_git.mjs evo lineage <directory>
```

### Bus Event Emission

All state-changing operations emit events to the Pluribus bus:

> **Data Layer Clarification:**
> - `$PLURIBUS_BUS_DIR/events.ndjson` - Operational bus (append-only audit trail)
> - `.pluribus/index/irkg/header_snapshot.json` - Derived metrics for UNIFORM panel display (updated every 5 min by `header_snapshot_updater.py`)
> - `.pluribus/dr/header_events.ndjson` - Disaster recovery ring (100-cap circular buffer for catastrophic recovery)
>
> Agents reading bus counts for the UNIFORM header should prefer the IRKG snapshot for efficiency; the DR ring serves as a fallback when the snapshot is stale or unavailable. See `nucleus/specs/UNIFORM.md` field defaults table.

| Command | Event Topic | Event Kind |
|---------|-------------|------------|
| `init` | `git.init` | `artifact` |
| `commit` | `git.commit` | `artifact` |
| `branch` | `git.branch` | `artifact` |
| `checkout` | `git.checkout` | `artifact` |
| `evo branch` | `git.evo.branch` | `artifact` |
| `evo hgt` | `git.evo.hgt.applied` | `artifact` |

---

## HGT Guard Ladder (G1-G6)

Every Horizontal Gene Transfer operation passes through six security gates:

### G1: Type Compatibility
```
Verifies the structural integrity of the source commit.
- Checks commit object format
- Validates tree structure
- Ensures blob integrity
```

### G2: Timing Compatibility
```
Prevents temporal paradoxes.
- Rejects commits with future timestamps
- Validates author/committer dates
- Ensures monotonic time progression
```

### G3: Effect Boundary
```
Protects Ring 0 (critical system files).
Protected paths include:
- nucleus/specs/*
- nucleus/tools/iso_git.mjs
- .claude/settings.json
- Any file marked with @RING0 annotation
```

### G4: Omega Acceptance
```
Validates lineage compatibility.
- Checks DAG ID ancestry
- Ensures evolutionary alignment
- Prevents orphan commits
```

### G5: MDL Penalty (Minimum Description Length)
```
Assesses change complexity.
- Measures diff size vs. baseline
- Flags unusually large changes
- Prevents "junk DNA" accumulation
```

### G6: Spectral Stability (Future)
```
Cryptographic provenance verification.
- Homomorphic Keyed Signatures (HKS)
- PQC (Post-Quantum Cryptography) readiness
- Commit signing validation
```

---

## git_server.py - HTTP API

### Endpoints

#### Git Operations

```http
GET /git/log?limit=50
GET /git/status
GET /git/branches
GET /git/show/<sha>
POST /git/hgt
  Body: { "source_ref": "<commit-sha>", "target_branch": "<branch>" }
```

#### Filesystem Operations

```http
GET /fs/tree?path=/pluribus&depth=2
GET /fs/read?path=/pluribus/README.md
POST /fs/write
  Body: { "path": "/path/to/file", "content": "..." }
```

#### SOTA (State of the Art) Operations

```http
GET /sota/papers?query=transformer+architecture
GET /sota/ingest?url=https://arxiv.org/abs/...
```

#### Module Operations

```http
GET /module/list
GET /module/info/<module-name>
POST /module/install
  Body: { "name": "<package>", "version": "latest" }
```

### Server Configuration

```python
# Default port
PORT = 9300

# CORS enabled for dashboard
CORS_ORIGINS = ["http://localhost:5173", "https://kroma.live"]

# Repository path
REPO_DIR = os.environ.get("PLURIBUS_REPO_DIR", "/pluribus")
```

---

## Lineage Tracking

### DAG IDs

Every evolutionary branch has a unique DAG ID:

```
Format: dag-<timestamp>-<random>
Example: dag-1734285600-x7k2m9
```

### Lineage JSON Structure

```json
{
  "dag_id": "dag-1734285600-x7k2m9",
  "parent_dag": "dag-1734200000-a1b2c3",
  "branch": "evo/20251215-feature",
  "created_at": "2025-12-15T18:00:00Z",
  "created_by": "claude-opus",
  "transfer_type": "VGT",
  "guard_results": {
    "G1": "pass",
    "G2": "pass",
    "G3": "pass",
    "G4": "pass",
    "G5": "pass",
    "G6": "skip"
  }
}
```

### Transfer Types

| Type | Name | Description |
|------|------|-------------|
| VGT | Vertical Gene Transfer | Standard parent→child branching |
| HGT | Horizontal Gene Transfer | Cross-lineage code transfer |

---

## Semantic Branch Naming (HOLON/ARK)

Dense semantic branch names encode intent and lineage for IRKG parsing.

Format:
```
evo/<YYYYMMDD>-<holon>.<domain>.<intent>.<surface>.<store>.<phase>.<actor>.<ark>
```

Example:
```
evo/20260126-holon.observability.plan.header.irkg.p0.codex.arks0.5
```

Spec: `nucleus/specs/holon_semantic_naming_v1.md`

## Dashboard Integration

### GitView Component

The dashboard's GitView (`/nucleus/dashboard/src/components/GitView.tsx`) provides:

1. **DAG Visualization**: Neon-styled commit graph
2. **Branch Selector**: Switch between evolutionary branches
3. **Commit Details**: Author, date, message, SHA
4. **HGT Trigger**: "🧬 HGT Push" button for controlled transfers
5. **Status Panel**: Working tree status and diff preview

### API Integration

```typescript
// GitView fetches from Caddy-proxied endpoints
const log = await fetch('/api/git/log?limit=50').then(r => r.json());
const status = await fetch('/api/git/status').then(r => r.json());
const branches = await fetch('/api/git/branches').then(r => r.json());
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PLURIBUS_BUS_DIR` | `.pluribus/bus` | Bus events directory |
| `PLURIBUS_ACTOR` | `iso_git_tool` | Actor name for bus events |
| `PLURIBUS_REPO_DIR` | `/pluribus` | Repository root |

---

## Sextet Compliance

The isomorphic-git integration follows the Pluribus Sextet model:

| Aspect | Implementation |
|--------|----------------|
| **Source** | Local filesystem (repository) |
| **Transducer** | `isomorphic-git` (FS ↔ Git objects) |
| **Memory** | `.git/` directory |
| **Feedback** | JSON events to bus |
| **Object** | Repository state |
| **Process** | Git operations (init, commit, branch, etc.) |
| **Observer** | Calling agent via `PLURIBUS_ACTOR` |

---

## Best Practices

### For Agents

1. **Always use iso_git.mjs** for Git operations (not native `git`)
2. **Set PLURIBUS_ACTOR** before operations for proper attribution
3. **Check guard results** after HGT operations
4. **Use evolutionary branches** for experimental changes

### For Operators

1. **Monitor bus events** for audit trail
2. **Review G3 violations** (Ring 0 protection)
3. **Track lineage DAG** for evolutionary history
4. **Use dashboard GitView** for visual exploration

### Branch Naming Convention

```
evo/<YYYYMMDD>-<description>
```

Examples:
- `evo/20251215-supermotd`
- `evo/20251214-telemetry-fix`
- `evo/20251213-hgt-guard`

---

## Related Documentation

- [Kroma Architecture](./kroma.md) - Overall system design
- [DKIN/CKIN Protocol](../specs/ckin_protocol_v16.md) - Dashboard kernel specification (v16)
- [Workflows](./workflows/) - Agent workflow guides
