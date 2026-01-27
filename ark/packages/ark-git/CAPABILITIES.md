# ark-git: Capability Map

**Source of Truth:** `nucleus/tools/iso_git.mjs`
**Target:** `@ark/git`

## Core Git Operations
| Function | Status | Implementation Plan |
|----------|--------|---------------------|
| `init` | Pending | Wrap `isomorphic-git.init` |
| `commit` | Pending | Wrap `isomorphic-git.commit` + PQC Sign |
| `commit-paths` | Pending | Partial add/remove + Commit |
| `status` | Pending | `statusMatrix` + Formatting |
| `log` | Pending | `git.log` -> JSON |
| `show` | Pending | `readCommit` + Tree Diff |
| `diff` | Pending | Tree comparison (affected projects) |
| `checkout` | Pending | `git.checkout` |
| `branch` | Pending | `git.branch` |

## Identity & Evolution (Genetic)
| Function | Status | Implementation Plan |
|----------|--------|---------------------|
| `evo branch` | Pending | VGT Lineage tracking (Span Context) |
| `evo hgt` | Pending | Cherry-pick with Guard Ladder (G1-G6) |
| `evo lineage` | Pending | Read `.pluribus/lineage.json` |
| `pqc sign` | Pending | HMAC-SHA256 / Dilithium stub |
| `verify` | Pending | Verify commit signature headers |

## Boundary (Native Git Fallback)
| Function | Status | Implementation Plan |
|----------|--------|---------------------|
| `push` | Pending | Spawn `git push` (Guarded) |
| `fetch` | Pending | Spawn `git fetch` (Guarded) |
| `clone` | Pending | Spawn `git clone` (Guarded) |

## Dependency Graph
- `isomorphic-git`
- `fs` (Node)
- `crypto` (Node)
- `@ark/spine` (for Lineage tracking integration later)
