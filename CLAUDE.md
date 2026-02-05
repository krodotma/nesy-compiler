# Pluribus Agent Instructions

## PBTSO Orchestration - CRITICAL RULES

### Authentication (NO API KEYS for main agents)

| Agent | Auth Method |
|-------|-------------|
| Codex | ChatGPT OAuth (browser login) |
| Claude | Anthropic OAuth |
| Gemini | Google OAuth (browser) |
| GLM | API key (free tier OK) |

### Manifest Rules

**DO NOT add `"model":` fields to swarm manifests.**

Correct:
```json
{"runner": "codex", "prompt": "..."}
{"runner": "claude", "prompt": "..."}
```

Wrong:
```json
{"runner": "codex", "model": "gpt-4o"}  // BREAKS - not on ChatGPT account
{"runner": "claude", "model": "claude-3-opus"}  // BREAKS - deprecated
```

### Default Models (automatic, don't override)

- Codex: `gpt-5.2-codex` with `xhigh` reasoning
- Claude: Current default from wrapper

### Key Files

- Orchestrator: `nucleus/tools/tmux_swarm_orchestrator.py`
- CLI: `nucleus/tools/pli.py`
- Manifests: `nucleus/manifests/*.json`
- Auth spec: `nucleus/specs/ORCHESTRATION_AUTH_FLOW.md`

### If Something Breaks

1. Check manifest for model overrides - REMOVE THEM
2. Check tmux pane output: `tmux capture-pane -t <session>:<window> -p`
3. Check prompt file: `ls .pluribus/swarm_prompts/`
