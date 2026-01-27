# ark-graph: Cortex Connector (FalkorDB)

**Version:** 1.0.0
**Status:** DRAFT
**Context:** The "Cognitive Core" handling semantic relationships and reasoning.

## Graph Schema (Cypher)

### Nodes
*   `(:Agent {id: string, kind: string})`
*   `(:Task {id: string, status: string})`
*   `(:Concept {name: string, embedding: float[]})`
*   `(:Artifact {hash: string, type: string})`

### Edges
*   `(:Agent)-[:OWNS]->(:Task)`
*   `(:Task)-[:PRODUCED]->(:Artifact)`
*   `(:Agent)-[:KNOWS]->(:Concept)`
*   `(:Concept)-[:RELATED_TO]->(:Concept)`

## Vector Indexing
FalkorDB supports vector indexing on node properties.
*   **Index:** `Concept.embedding` (HNSW)
*   **Query:** `CALL db.idx.vector.queryNodes('Concept', 'embedding', $vec, 5)`

## Interface (`@ark/graph`)
```typescript
interface GraphDriver {
  query(cypher: string, params?: object): Promise<Result>;
  vectorSearch(label: string, prop: string, vector: number[], k: int): Promise<Node[]>;
}
```

## Migration from `falkordb_api.py`
The existing Python API is a thin wrapper. The Node.js implementation will use the native Redis client (FalkorDB uses the Redis protocol) for higher performance.
