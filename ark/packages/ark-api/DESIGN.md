# ark-api: Fastify Architecture

**Version:** 1.0.0
**Status:** DRAFT
**Context:** The "Membrane" (Ingress/Egress) layer.

## Architecture
We use **Fastify** for its low overhead and robust plugin ecosystem.

### Core Plugins
1.  **`ark-context`**: Injects `spine`, `graph`, and `bus` instances into `request.ark`.
2.  **`ark-auth`**: Membrane security (JWT/PQC signature verification).
3.  **`ark-schema`**: Shared JSON Schemas (Zod/Ajv) for validation.

## Routes (REST)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/rpc/:procedure` | Direct method call (e.g., `git.commit`). |
| POST | `/v1/bus/emit` | Publish event to Bus. |
| GET | `/v1/state/:key` | Read from Spine (Key/Value). |
| POST | `/v1/graph/query` | Cypher query to Cortex. |
| GET | `/health` | Liveness check. |

## Inter-Process Communication (IPC)
`ark-api` serves as the HTTP gateway for local tools (CLI) and the React Frontend. It communicates with the `ark-bus` via internal Node.js `EventEmitter` or local socket.
