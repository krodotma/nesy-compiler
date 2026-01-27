# ark-spine: Schema Definition (LMDB)

**Version:** 1.0.0
**Status:** DRAFT
**Context:** The "Spine" is the authoritative, append-only ledger of state.

## Databases (DBI)

LMDB supports multiple named databases within a single environment. We partition data to ensure locality and scan performance.

### 1. `registry` (The Neurosymbolic Spine)
*   **Key:** `agent_id` (string)
*   **Value:** JSON (Compressed)
*   **Schema:**
    ```json
    {
      "id": "gemini",
      "kind": "LLM",
      "manifest_sha": "sha256...",
      "state": "active",
      "heartbeat_ts": 1735689000,
      "capabilities": ["code", "vision"],
      "lineage": {
        "parent": "opus-1",
        "generation": 5
      }
    }
    ```

### 2. `task_ledger` (DKIN State)
*   **Key:** `task_id` (UUID)
*   **Value:** JSON (Compressed)
*   **Schema:**
    ```json
    {
      "id": "uuid-...",
      "status": "in_progress",
      "owner": "gemini",
      "title": "Refactor Spine",
      "created_ts": 1735689000,
      "updated_ts": 1735689100,
      "artifacts": ["sha256:blob1", "sha256:blob2"]
    }
    ```

### 3. `bus_events` (Transient Buffer)
*   **Key:** `timestamp_ns:event_id` (Lexicographical sort)
*   **Value:** MessagePack (Compact)
*   **Retention:** Capped (e.g., last 1M events). Archived to MinIO.

### 4. `system_metrics` (Telemetry)
*   **Key:** `metric_name:timestamp`
*   **Value:** Float64 / Struct
*   **Purpose:** Dashboard rendering (SuperMOTD).

## Access Patterns
*   **Reads:** Direct key lookup or Range Scan (Cursor).
*   **Writes:** Single-writer thread (Node.js main loop).
*   **Durability:** Async sync (nosync) for Bus, Sync for Registry/Ledger.

## Integration
*   **Library:** `lmdb-js` (via `node-lmdb`).
*   **Interface:** `ark-spine` package exports typed accessors.
