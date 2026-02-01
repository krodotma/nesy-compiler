# Dialogos Protocol v1 - Agent Communication Layer

**Version:** v1.0 | **Authority:** DNA.md Principle #1 (Sovereignty)

---

## Purpose

Dialogos owns all agent-to-agent and agent-to-LLM communication. No direct API calls; all requests flow through the unified bus.

## Message Schema

### Request
```json
{
  "id": "req-uuid-v4",
  "ts": 1735574400.123,
  "iso": "2025-12-30T12:00:00.123Z",
  "topic": "dialogos.submit.claude",
  "kind": "request",
  "level": "info",
  "actor": "codex_codemaster",
  "data": {
    "req_id": "req-uuid-v4",
    "provider": "claude-opus",
    "prompt": "Analyze the following code...",
    "context": {
      "files": ["nucleus/tools/agent_bus.py"],
      "task_id": "task-uuid"
    },
    "timeout_s": 120
  }
}
```

### Response
```json
{
  "id": "res-uuid-v4",
  "ts": 1735574500.456,
  "iso": "2025-12-30T12:01:40.456Z",
  "topic": "dialogos.cell.claude",
  "kind": "response",
  "level": "info",
  "actor": "dialogos",
  "data": {
    "req_id": "req-uuid-v4",
    "provider": "claude-opus",
    "success": true,
    "response": "The code implements...",
    "usage": {
      "prompt_tokens": 1234,
      "completion_tokens": 567
    }
  }
}
```

## Provider Abstraction

| Provider ID | Model | Rate Limit | Priority |
|-------------|-------|------------|----------|
| `claude-opus` | Claude Opus | 10 rpm | 1 (highest) |
| `claude-sonnet` | Claude Sonnet | 50 rpm | 2 |
| `gemini-2` | Gemini 2.0 | 60 rpm | 2 |
| `kimi-2-5` | Kimi 2.5 | 60 rpm | 3 |
| `qwen-plus` | Qwen | 100 rpm | 3 |

## Request/Response Correlation

- Each request has unique `req_id`
- Response includes matching `req_id` in data
- Non-blocking: requestor monitors bus for response
- Timeout: requestor emits `dialogos.timeout` if TTL exceeded

## Bus Topics

| Pattern | Direction | Purpose |
|---------|-----------|---------|
| `dialogos.submit.*` | Agent → Bus | Request to LLM |
| `dialogos.cell.*` | Bus → Agent | Response from LLM |
| `dialogos.error.*` | Bus → Agent | Error notification |
| `dialogos.timeout` | Agent → Bus | Request timeout |
