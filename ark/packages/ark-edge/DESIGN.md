# ark-edge: uWebSockets.js Design

**Version:** 1.0.0
**Status:** DRAFT
**Context:** High-performance WebSocket layer for real-time telemetry and "hot" data.

## Why uWebSockets.js?
Standard Node.js `ws` or `socket.io` struggle with massive broadcast fanout (e.g., 10k dashboard elements updating at 60fps). `uWebSockets.js` (C++ binding) provides order-of-magnitude better throughput.

## Pub/Sub Channels

| Channel | Frequency | Data |
|---------|-----------|------|
| `sys/metrics` | 1Hz | CPU, RAM, Loop Lag (SuperMOTD). |
| `bus/broadcast` | Real-time | Raw A2A Bus events. |
| `task/:id` | Event-driven | Task progress updates. |
| `vision/stream` | 30fps | MJPEG/h264 chunks from `ark-vision`. |

## Architecture
*   **Gateway:** Runs on port 4000 (distinct from API port 3000).
*   **Bridge:** Subscribes to `ark-bus` and rebroadcasts to WebSocket topics.
*   **Security:** Token-based handshake (verified against `ark-api` auth).

## WebLLM Integration
`ark-edge` will optionally host the `WebLLM` service worker coordination, allowing browser-based inference nodes to register as compute resources.
