# Theia Vision Protocol v1.0

**Date:** 2026-01-24  
**Status:** Draft

---

## Overview

Protocol specification for Theia vision capture and browser automation.

---

## Endpoints

### POST /v1/theia/ingest

Ingest frames from VisionEye dashboard.

```json
{
  "frames": ["base64...", "base64..."],
  "timestamp": 1737772800.0,
  "meta": {
    "fps": 1.0,
    "source": "screen",
    "resolution": [1920, 1080]
  }
}
```

Response:
```json
{
  "count": 60,
  "buffer_size": 60
}
```

---

### POST /v1/theia/act

Execute browser action.

```json
{
  "action": "fill",
  "selector": "textarea",
  "value": "Hello world",
  "headers": {"Authorization": "Bearer token"}
}
```

Actions: `open`, `fill`, `click`, `press`, `close`

Response:
```json
{
  "success": true,
  "message": "filled"
}
```

---

### POST /v1/theia/chat

Send message to webchat provider.

```json
{
  "provider": "claude",
  "message": "Hello!",
  "auth_token": "bearer-token",
  "wait_seconds": 3.0
}
```

Providers: `claude`, `chatgpt`, `gemini`

Response:
```json
{
  "snapshot": "accessibility-tree-text",
  "provider": "claude",
  "latency_ms": 3500.0
}
```

---

### POST /v1/theia/infer

VLM inference on image.

```json
{
  "image": "base64...",
  "prompt": "What do you see?",
  "model": "llava:7b",
  "use_icl": true
}
```

Response:
```json
{
  "content": "This is a screenshot of...",
  "model": "llava:7b",
  "latency_ms": 1200.0,
  "tokens_used": 150
}
```

---

## Ring Buffer Format

Dashboard VisionEye captures frames to ring buffer:

```typescript
interface Frame {
  ts: number;       // Unix timestamp ms
  data: string;     // Base64 JPEG
  meta?: {
    width: number;
    height: number;
    quality: number;
  };
}
```

Buffer config:
- **Size:** 60 frames (1 minute at 1 FPS)
- **Quality:** JPEG 70%
- **Scale:** 50% of original

---

## Selectors

CSS selectors for webchat providers:

| Provider | Input Selector | Submit |
|----------|---------------|--------|
| Claude | `div[contenteditable], textarea` | Enter |
| ChatGPT | `#prompt-textarea` | Enter |
| Gemini | `textarea` | Enter |

---

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad request (missing required field) |
| 401 | Auth token invalid/expired |
| 500 | Browser action failed |
| 503 | agent-browser not available |
