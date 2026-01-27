# ark-storage: MinIO Binding Layer

**Version:** 1.0.0
**Status:** DRAFT
**Context:** Blob storage for "Artifacts" (Video, Audio, Logs, Large JSON).

## Buckets

| Bucket | Access | Retention | Purpose |
|--------|--------|-----------|---------|
| `artifacts` | Private | Forever | Build artifacts, task outputs. |
| `perception` | Private | 7 Days | Raw video streams (Theia), Audio (Auralux). |
| `memory` | Private | Forever | Crystallized memories (RAG docs). |
| `public` | Public | Forever | UI Assets, Avatars. |
| `archive` | Cold | Forever | Rotated logs, old git packs. |

## Path Strategy (CAS)
We use Content Addressable Storage (CAS) for immutability where possible.
*   **Format:** `sha256/{hash[0:2]}/{hash}`
*   **Benefits:** De-duplication, integrity verification.

## Interface (`@ark/storage`)
```typescript
interface StorageDriver {
  put(bucket: string, key: string, stream: Readable): Promise<string>; // returns ETag
  get(bucket: string, key: string): Promise<Readable>;
  stat(bucket: string, key: string): Promise<Metadata>;
  list(bucket: string, prefix: string): AsyncIterable<Entry>;
}
```

## Local Dev Fallback
If MinIO is unavailable, `ark-storage` falls back to `fs` (FileSystem) mode in `.pluribus/storage`.
