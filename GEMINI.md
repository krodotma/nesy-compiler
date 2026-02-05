# GEMINI AGENT INSTRUCTIONS FOR PLURIBUS

## STOP - READ THIS BEFORE MODIFYING ORCHESTRATION

### Authentication - NO API KEYS FOR MAIN AGENTS

**Codex uses ChatGPT OAUTH, not API keys.**
**Claude uses Anthropic OAUTH, not API keys.**
**You (Gemini) use Google OAUTH, not API keys.**

Only GLM uses API keys (free tier).

### MANIFEST RULES - CRITICAL

When creating or modifying swarm manifests in `nucleus/manifests/`:

**NEVER add `"model":` fields.** They break things.

#### CORRECT:
```json
{
  "id": "codex-worker",
  "runner": "codex",
  "prompt": "Your task"
}
```

#### WRONG (DO NOT DO THIS):
```json
{
  "id": "codex-worker",
  "model": "gpt-4o",      // WRONG - fails on ChatGPT account
  "model": "o1",          // WRONG - fails
  "model": "claude-3-opus" // WRONG - deprecated 2024 model
}
```

### Default Models (AUTOMATIC - DON'T OVERRIDE)

- **Codex**: `gpt-5.2-codex` with `xhigh` reasoning (from `~/.codex/config.toml`)
- **Claude**: Current default from `bus-claude` wrapper
- **Gemini**: Your default model

### WHY THIS MATTERS

Previous failures:
1. You added `"model": "claude-3-opus"` → 404 error (deprecated)
2. You added `"model": "gpt-4o"` → "not supported on ChatGPT account"
3. You added `"model": "o1"` → same error

The orchestration WORKS when you don't override models.

### Files You Should Know

- `nucleus/tools/tmux_swarm_orchestrator.py` - Spawns agents in tmux
- `nucleus/tools/pli.py` - CLI for PBTSO operations
- `nucleus/manifests/*.json` - Swarm definitions (NO MODEL FIELDS)
- `nucleus/specs/ORCHESTRATION_AUTH_FLOW.md` - Full auth documentation

### Testing a Swarm

```bash
# Spawn
python3 nucleus/tools/pli.py swarm nucleus/manifests/tooling_test.json

# Check status
python3 nucleus/tools/pli.py a2a --status --json

# View agent output
tmux capture-pane -t pluribus_swarm:<agent_id> -p
```

### If You Break Something

1. Check the manifest for `"model":` fields - DELETE THEM
2. Verify the runner field is set: `"runner": "codex"` or `"runner": "claude"`
3. Don't guess model names - just remove the model field entirely

---

## GEMINI FAQ - ANSWERS TO YOUR QUESTIONS

### Q1: Runner Specificity - How to use "Claude 4.5 Opus UltraThink"?

**Answer: Use `"runner": "claude"` - the bus-claude wrapper handles model selection.**

The RUNNER_CONFIG maps runners to wrappers:
- `"runner": "claude"` → `bus-claude` wrapper → default model (currently claude-4.5-opus)
- `"runner": "codex"` → `bus-codex` wrapper → default model (gpt-5.2-codex)

**DO NOT** create new runners like `"runner": "ag-opus"`. The wrapper handles it.

```json
// CORRECT - bus-claude uses the best available model
{
  "id": "opus-analyst",
  "runner": "claude",
  "prompt": "Deep analysis task"
}
```

### Q2: Reasoning Effort - How to get "xhigh" for Codex?

**Answer: System-wide config in `~/.codex/config.toml` controls reasoning effort.**

```toml
# ~/.codex/config.toml
model = "gpt-5.2-codex"
model_reasoning_effort = "xhigh"
```

**There is NO per-swarm override.** All codex agents inherit the system config.

If you need different reasoning levels for different tasks, the system admin must change the config.toml. Do NOT try to add `"reasoning": "xhigh"` to manifests - it won't work.

### Q3: Are Ultra models pre-configured?

**Yes.** The environment is pre-configured:
- Codex: gpt-5.2-codex with xhigh reasoning (from config.toml)
- Claude: claude-4.5-opus (from bus-claude wrapper default)
- Gemini: Your default model (from browser OAuth session)

### PROCEED WITH SWARMS

You are cleared to execute swarms using:
- `"runner": "claude"` for Claude tasks
- `"runner": "codex"` for Codex tasks
- `"runner": "gemini"` for your tasks
- `"runner": "glm"` for GLM tasks (only one that uses API keys)

**GO.** Execute your swarms. The environment is ready.
