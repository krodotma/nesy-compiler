# PBTSO ORCHESTRATION AUTHENTICATION FLOW
**Status: CANONICAL - DO NOT MODIFY WITHOUT CONSENSUS**
**Last Updated: 2026-02-05**

## CRITICAL: READ THIS BEFORE TOUCHING ORCHESTRATION CODE

### Authentication Methods by Runner

| Runner | Auth Method | Config Location | Notes |
|--------|-------------|-----------------|-------|
| **codex** | ChatGPT Account (OAuth) | `~/.codex/` | NO API KEYS. Uses browser OAuth flow. |
| **claude** | Anthropic Account (OAuth) | `~/.claude/` | NO API KEYS. Uses OAuth flow. |
| **glm** | API Key (Free Tier) | `~/.config/agent-wrapper/keys` | Free tier, key-based auth OK |
| **gemini** | Google Account (OAuth) | Browser session | NO API KEYS in orchestration |

### CODEX SPECIFICS

**Codex uses ChatGPT account authentication, NOT API keys.**

The default model from `~/.codex/config.toml`:
```toml
model = "gpt-5.2-codex"
model_reasoning_effort = "xhigh"
```

**DO NOT specify models in manifests unless you have verified they work with ChatGPT accounts.**

Models that DON'T work with ChatGPT account:
- `gpt-4o` - NOT SUPPORTED
- `o1` - NOT SUPPORTED
- `o3-mini` - NOT SUPPORTED

Models that WORK:
- `gpt-5.2-codex` (default) - WORKS
- No model specified (uses default) - WORKS

### MANIFEST RULES

1. **DO NOT add `"model":` fields to manifests** unless absolutely necessary
2. If you must specify a model, verify it first with: `codex --model <name> -p "test"`
3. Codex agents should have `"runner": "codex"` with NO model override
4. Claude agents should have `"runner": "claude"` with NO model override

### CORRECT MANIFEST EXAMPLE

```json
{
  "agents": [
    {
      "id": "codex-worker-1",
      "role": "Engineer",
      "runner": "codex",
      "prompt": "Your task here",
      "ring": 2
    },
    {
      "id": "claude-analyst-1",
      "role": "Analyst",
      "runner": "claude",
      "prompt": "Your task here",
      "ring": 2
    }
  ]
}
```

### INCORRECT (DO NOT DO THIS)

```json
{
  "agents": [
    {
      "id": "codex-worker-1",
      "model": "gpt-4o",        // WRONG - will fail
      "model": "o1",            // WRONG - will fail
      "model": "claude-3-opus", // WRONG - deprecated
    }
  ]
}
```

### THE BUS-CODEX WRAPPER

`/pluribus/nucleus/tools/bus-codex` handles:
1. Setting up agent home directory
2. Loading XDG paths
3. Emitting bus events for observability
4. Delegating to the actual `codex` binary

It does NOT handle API key auth for codex - codex uses its own OAuth flow.

### PROMPT PASSING

Large prompts (>500 chars) are written to temp files:
- Location: `/pluribus/.pluribus/swarm_prompts/<agent_id>_<timestamp>.prompt`
- Passed via stdin: `codex exec ... - < /path/to/prompt.file`

This avoids shell escaping issues with tmux send-keys.

### TROUBLESHOOTING

**Error: "model not supported when using Codex with a ChatGPT account"**
- Remove the `"model":` field from the manifest
- Let codex use its default (gpt-5.2-codex)

**Error: "model: claude-3-opus" 404**
- This model is deprecated (it's from 2024)
- Remove the `"model":` field, use default

**Agent exits immediately in tmux**
- Check if prompt file was written to `.pluribus/swarm_prompts/`
- Verify stdin redirect is working: `< /path/to/file`

---

## FOR GEMINI SPECIFICALLY

Dear Gemini,

When working on PBTSO orchestration:

1. **NEVER add model overrides to manifests** - they break things
2. **NEVER use API keys for codex** - it uses ChatGPT OAuth
3. **ALWAYS use `"runner": "codex"` or `"runner": "claude"`** without model fields
4. **CHECK existing working manifests** before creating new ones

The orchestration was working fine until model overrides were added. Keep it simple.

Sincerely,
The Pluribus System
