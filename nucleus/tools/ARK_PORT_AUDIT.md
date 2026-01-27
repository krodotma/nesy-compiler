# Audit: nucleus/tools -> Ark Port

**Date:** 2026-01-27
**Scope:** 471 Python scripts in `nucleus/tools`.
**Target:** Node.js (Ark) Monorepo.

## Classification Legend
- **[D] Daemon:** Long-running service (needs `ark-service` or Docker).
- **[L] Logic:** Pure business logic (port to `ark-core` lib).
- **[C] CLI:** Command-line tool (port to `ark-cli`).
- **[X] Deprecate:** Legacy or unused (move to `archive/`).
- **[K] Keep:** Python specific (e.g., PyTorch/ML) - Keep as Python microservice.

## Audit Log

| File | Category | Notes |
|------|----------|-------|
| `a2a_adapter.py` | [L] | Core A2A logic. Port to `ark-a2a`. |
| `a2a_bridge.py` | [D] | Bridge daemon. Port to `ark-bridge`. |
| `agent_bus.py` | [L] | Core Bus logic. Port to `ark-bus`. |
| `agent_header.py` | [C] | REPL Header. Port to `ark-cli` (or keep as polyglot). |
| `ckin_report.py` | [C] | Reporting tool. Port to `ark-cli`. |
| `iso_git.mjs` | [L] | Already Node.js. Move to `ark-git`. |
| `iso_pqc.mjs` | [L] | Already Node.js. Move to `ark-git`. |
| `log_hygiene.py` | [D/C] | System maintenance. Port to `ark-ops`. |
| `omega_dispatcher.py` | [D] | Core Dispatcher. Port to `ark-omega`. |
| `supermotd.py` | [C] | Dashboard. Port to `ark-ui` / `ark-cli`. |

## Bulk Analysis (Patterns)

### 1. Daemons (to Ark Services)
- `*_daemon.py`
- `bus_mirror_daemon.py`
- `browser_session_daemon.py`
- `metaingest_daemon.py`
- `reconciler_daemon.py`

### 2. Operators (to Ark Ops)
- `*_operator.py`
- `pblock_operator.py`
- `pbflush_operator.py`
- `pbskills_operator.py`

### 3. ML/AI (Keep Python)
- `agent_lightning.py` (PyTorch)
- `embeddings.py` (SentenceTransformers)
- `rag_vector.py` (Chroma/FAISS)
- `sota_*.py` (Research/ML)

### 4. Tests
- `tests/*` (Port logic, use Vitest for Ark).

## Migration Priority
1. **Bus & Git:** `agent_bus` + `iso_git` (The Spine).
2. **State:** `dkin_ledger` + `falkordb_api` (The Memory).
3. **Dispatch:** `omega_dispatcher` + `a2a_adapter` (The Brain).
