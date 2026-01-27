# HOLON Semantic Naming v1
# Status: DRAFT
# Date: 2026-01-26

This specification defines **dense, semantically rich naming conventions** for
branches and agents. The goal is to encode intent, scope, and provenance directly
into identifiers so they can be parsed into IRKG nodes/edges and aligned with
ARK/entelexis evolution.

## 1) Design Goals
- **Semantic density**: names encode who/what/why/where/when.
- **Parseable**: deterministic tokenization into IRKG properties.
- **Stable**: minimal churn across iterations.
- **ARK-aligned**: embed CMP or ARK tag for lineage fitness.
- **IsoGit-compatible**: safe for refs and filenames.

## 2) Branch Naming (Evolutionary)

### 2.1 Format
```
evo/<YYYYMMDD>-<holon>.<domain>.<intent>.<surface>.<store>.<phase>.<actor>.<ark>
```

### 2.2 Field Semantics
- **YYYYMMDD**: start date of the workstream.
- **holon**: top-level system context.
  - Allowed: holon | pluribus | omega | dialogos | ark | irkg
- **domain**: semantic focus area.
  - Allowed: protocol | observability | coordination | evolution | learning | safety
- **intent**: why this branch exists.
  - Allowed: plan | spec | migrate | impl | verify | audit
- **surface**: subsystem or interface targeted.
  - Allowed: header | eventstore | bus | portal | pipeline | recovery | ui
- **store**: evidence plane strategy.
  - Allowed: irkg | dr | hybrid
- **phase**: lifecycle stage.
  - Allowed: p0 | p1 | p2 | p3 | p4
- **actor**: owning agent.
  - Allowed: codex | claude | gemini | qwen | grok | multi
- **ark**: ARK or CMP tag (short, normalize dots to hyphens).
  - Pattern: arks<cmp> or arkm<cmp> (example: arks0-5)

### 2.3 Example
```
evo/20260126-holon.observability.plan.header.irkg.p0.codex.arks0-5
```
Note: ARK tag `0.5` normalized to `0-5` to avoid delimiter ambiguity.

## 3) Agent Semantic ID

### 3.1 Format
```
<class>.<role>.<domain>.<focus>.<lane>.r<ring>.<model>.<variant>
```

### 3.2 Field Semantics
- **class**: agent class.
  - Allowed: sagent | swagent | cagent
- **role**: functional role.
  - Allowed: planner | architect | auditor | engineer | operator | verifier
- **domain**: primary system context.
  - Allowed: holon | pluribus | irkg | dialogos | omega
- **focus**: task focus.
  - Allowed: header | eventstore | protocol | migration | recovery
- **lane**: operational lane.
  - Allowed: dialogos | ops | qa | sys
- **ring**: control ring.
  - Allowed: r0 | r1 | r2 | r3
- **model**: model family (normalize version dots to hyphens; see 5.1).
  - Examples: claude-opus-4-5 | gpt-5-2 | gemini-3-pro | qwen-plus
- **variant**: execution mode.
  - Allowed: ultrathink | fast | safe | audit

### 3.3 Example
```
sagent.planner.holon.header.dialogos.r0.claude-opus-4-5.ultrathink
```
Note: Model version `4.5` normalized to `4-5` per rule 5.1.

## 4) IRKG Mapping (Suggested)
- **Branch** becomes a `evo_branch` node with properties:
  - date, holon, domain, intent, surface, store, phase, actor, ark
- **Agent ID** becomes `agent_semantic_id` on `agent` nodes.

## 5) Validation Rules
- Lowercase only; hyphen and dot separators.
- Max length 120 chars (truncate tail if needed).
- No spaces, slashes only for `evo/` prefix.

### 5.1 Dot-in-Model-Name Rule

**Problem:** Model names often contain version dots (e.g., `claude-opus-4.5`, `gpt-5.2`).
Since `.` is the field delimiter, this creates parsing ambiguity.

**Rule:** Replace dots in model names with hyphens when used in semantic identifiers.

| Raw Model Name     | Normalized for ID   |
|--------------------|---------------------|
| `claude-opus-4.5`  | `claude-opus-4-5`   |
| `gpt-5.2`          | `gpt-5-2`           |
| `gemini-3.0-pro`   | `gemini-3-0-pro`    |
| `qwen-2.5-plus`    | `qwen-2-5-plus`     |

**Rationale:** Hyphens are already used within field values (e.g., `claude-opus`).
Normalizing version dots to hyphens ensures deterministic tokenization while
preserving readability.

### 5.2 Correct vs. Incorrect Examples

**Branch Names:**
```
# CORRECT: all 8 fields, no embedded dots
evo/20260126-holon.observability.plan.header.irkg.p0.codex.arks0-5

# INCORRECT: missing fields (only 6 after date)
evo/20260126-holon.irkg.header.plan.orch.codex

# INCORRECT: model version dot creates 9 fields
evo/20260126-holon.observability.plan.header.irkg.p0.claude-opus-4.5.arks0.5
```

**Agent IDs:**
```
# CORRECT: model version dot normalized to hyphen
sagent.planner.holon.header.dialogos.r0.claude-opus-4-5.ultrathink

# INCORRECT: version dot creates parsing ambiguity (9 fields instead of 8)
sagent.planner.holon.header.dialogos.r0.claude-opus-4.5.ultrathink

# INCORRECT: ark tag dot creates ambiguity
sagent.planner.holon.header.dialogos.r0.codex.arks0.5
```

## 6) Backward Compatibility
- Legacy branch names remain valid.
- New workstreams SHOULD adopt semantic naming for IRKG parse.
