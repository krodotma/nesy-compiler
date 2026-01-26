# REPL Header Contract v2.1 (PLURIBUS v1)

Status: ACTIVE
Scope: All Citizen agents

## Prime Directive

Before emitting the panel, perform ONE local **snapshot read**:
- Read IRKG header snapshot (single file read)
- Fallback: read DR ring (100‑cap NDJSON) if snapshot missing or stale

Forbidden before the panel:
- Network calls
- Full bus log scans or topic directory walks
- Long tool chains

### Snapshot Paths (defaults)
- IRKG snapshot: `/pluribus/.pluribus/index/irkg/header_snapshot.json`
  - Override: `PLURIBUS_HEADER_SNAPSHOT`
- DR ring (catastrophic only): `/pluribus/.pluribus/dr/header_events.ndjson`
  - Override: `PLURIBUS_HEADER_DR_RING`

### NDJSON Deprecation
NDJSON bus tails are deprecated for header metrics. Only the DR ring remains
as a bounded fallback (100 entries max).

## Default Panel (emit first in every response)

```
+------------------------------------------------------------------------------+
| UNIFORM v2.1 | <agent> | sess:new | cell:dia.1.0 | lane:dialogos | d:0       |
+------------------------------------------------------------------------------+
| PROTO | PLURIBUS v1 | gen:? | lineage:?                                      |
| SCORE | ########## 100/100 v | CDI:? | health:nominal                         |
| STATE | idle | hexis:? | bus:+0 | uncommit:? | branch:?                       |
| SWARM | SAGENT:? | CAGENT:? | active:?                                        |
| TASKS | #####..... 45% | active ?/? | task:"unknown"                          |
| BUS A2A | for:0 col:0 int:0 neg:0 task:0                                      |
| BUS OPS | pbtest:0 pblock:0 pbresume:0 pblive:0 pbflush:0 pbiud:0 pbcli:0 hyg:0
| BUS QA | alert:0 anom:0 rem:0 verd:0 live:0 lchk:0 act:0 hyg:0                |
| BUS SYS | tel:0 task:0 agent:0 dlg:0 ohm:0 omg:0 prov:0 dash:0 brow:0         |
| SCOPE | goal:"awaiting task" | artifact:none                                 |
+------------------------------------------------------------------------------+
```

`?` is valid for unknown values. Use it when data is unavailable.

## Tablet Panel (optional)

Set `UNIFORM_PANEL_STYLE=tablet` (or `badge`/`compact`) to emit a compact panel.
Minimum required fields:
- UNIFORM v2.1
- PLURIBUS v1
- agent id + lane
- task progress summary
- bus summary

## Canonical Versions

- PLURIBUS v1
- UNIFORM v2.1
- DKIN v30
- PAIP v16
- CITIZEN v2

## Bus Topics (canonical groups)

**A2A**
- a2a.design.forensics
- a2a.design.collaboration
- a2a.design.integration
- a2a.negotiate
- a2a.task.dispatch

**OPS**
- operator.pbtest
- operator.pblock
- operator.pbresume
- operator.pblive
- operator.pbflush
- operator.pbiud
- operator.pbclitest
- operator.hygiene

**QA**
- qa.alert
- qa.anomaly
- qa.remediation
- qa.verdict
- qa.live
- qa.live-checker
- qa.action
- qa.hygiene

**SYS**
- telemetry.client.error
- task.task_ledger.append
- agent.*
- dialogos.*
- ohm.*
- omega.*
- providers.*
- dashboard.*
- browser.*
