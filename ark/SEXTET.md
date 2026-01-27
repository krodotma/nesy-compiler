# The Pluribus Sextet: Bounded Contexts

**Version:** 1.0.0
**Status:** DRAFT
**Context:** Pluribus v2 (Hyperscale/Ark) Architecture

The "Sextet" defines the six fundamental bounded contexts of the Pluribus organism. Each context owns specific data, logic, and services.

## 1. Genetic (Identity & Evolution)
*   **Role:** Maintains the "code is law" invariant, cryptographic identity, and evolutionary history.
*   **Components:** `ark-git`, `iso_git`, `iso_pqc`, `ark-evolution`.
*   **Data:** Git ODB (Objects), Lineage JSON, PQC Keys.
*   **Package:** `@ark/git`

## 2. Spine (State & Truth)
*   **Role:** The high-performance, atomic, append-only ledger of system state.
*   **Components:** `ark-spine`, `dkin`, `lmdb`.
*   **Data:** LMDB K/V Store (Registry, Task Ledger, Manifests).
*   **Package:** `@ark/spine`

## 3. Nucleus (Orchestration & Bus)
*   **Role:** The central nervous system handling event routing, dispatch, and process lifecycle.
*   **Components:** `ark-bus`, `ark-omega` (Dispatcher), `ark-service` (Daemons).
*   **Data:** A2A Bus (NDJSON/FalkorDB), Process Table.
*   **Package:** `@ark/bus`, `@ark/nucleus`

## 4. Cortex (Cognition & Graph)
*   **Role:** Semantic reasoning, knowledge graph, and neurosymbolic integration.
*   **Components:** `ark-graph`, `ark-inference` (LLM Bridge).
*   **Data:** FalkorDB (Knowledge Graph), Vector Indices.
*   **Package:** `@ark/graph`

## 5. Membrane (Interface & Perception)
*   **Role:** The boundary between Self and Other. Handles API ingress, UI serving, and sensor inputs.
*   **Components:** `ark-api` (Fastify), `ark-edge` (uWebSockets), `vision_eye`.
*   **Data:** API Schemas, UI Assets, Auth Tokens.
*   **Package:** `@ark/api`, `@ark/edge`

## 6. Artifact (Storage & Memory)
*   **Role:** Long-term storage of blobs, media, and crystallized memories.
*   **Components:** `ark-storage`, `minio`, `ingest`.
*   **Data:** MinIO Buckets (Video, Audio, Logs, Backups).
*   **Package:** `@ark/storage`
