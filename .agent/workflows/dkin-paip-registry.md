---
description: DKIN and PAIP Protocol Version Registry (CANONICAL)
---

# DKIN/PAIP Protocol Registry

**CRITICAL**: When instructed to update DKIN or PAIP, ALWAYS re-read the latest spec files listed below from the VPS (`pluribus/remote_pluribus/nucleus/specs/`).

## DKIN (Latest: v25)

**Active Spec**: `nucleus/specs/dkin_protocol_v25_lifecycle.md`
**Status**: ACTIVE (Adopted 2025-12-23)

### Version Evolution
| Version | Codename | Spec File |
|---------|----------|-----------|
| v18 | Resilience | `nucleus/specs/dkin_protocol_v18_resilience.md` |
| v19 | Evolution | `nucleus/specs/dkin_protocol_v19_evolution.md` |
| v20 | Edge/Body | `nucleus/specs/dkin_protocol_v20_edge_body.md` |
| v21 | Lossless Handoff | `nucleus/specs/dkin_protocol_v21_lossless.md` |
| v22 | Amber | `nucleus/specs/dkin_protocol_v22_amber.md` |
| v23 | Lanes | `nucleus/specs/dkin_protocol_v23_lanes.md` |
| v24 | Verification | `nucleus/specs/dkin_protocol_v24_verification.md` |
| **v25** | **3-Tier Lifecycle** | `nucleus/specs/dkin_protocol_v25_lifecycle.md` |

## PAIP (Latest: v13.1, under CKIN)

**Active Spec**: `nucleus/specs/ckin_protocol_v13_phenomenology.md`
**Includes**: PAIP v13 (port/display/browser-context isolation) + v13.1 (zombie reaping/GC)

### Version Evolution
| Version | Codename | Spec File |
|---------|----------|-----------|
| v12 | Filesystem Isolation | `nucleus/specs/ckin_protocol_v12.md` |
| v12.1 | TTL + Summary Frame | `nucleus/specs/ckin_protocol_v12.1.md` |
| **v13** | **Phenomenology** | `nucleus/specs/ckin_protocol_v13_phenomenology.md` |
| v13.1 | Lifecycle Reaper/GC | (same file as v13) |

## Operator Registry

**Canonical Source**: `nucleus/specs/semops.json`
- Ties DKIN/PAIP versions to live operators.
- Check this file for topic definitions and schema requirements.

## Refresh Protocol

When user says "update DKIN" or "update PAIP":
1. Read the LATEST spec file from `pluribus/remote_pluribus/nucleus/specs/`.
2. Cross-reference with `semops.json` for operator mappings.
3. Update local tools/config to align with the latest version.
4. Re-publish audit report with corrected version numbers.
