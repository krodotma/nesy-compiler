/**
 * GRAPH-EMIT Stage Extension
 *
 * Step 9: Integrate Graph-Emit into compilation pipeline.
 * Replaces/augments JSON output with FalkorDB.merge(holon).
 *
 * This module extends the emit stage to:
 * - Store compiled holons in FalkorDB
 * - Create semantic relationships from LSA
 * - Update provenance lineage in the graph
 */

import type { CompilationContext } from '@nesy/core';
import type { CompiledIR, CompiledArtifact } from '../ir';
import type { EmitOutput, StageHistoryEntry, IntentMetadata } from './emit';

// FalkorDB types (from @nesy/persistence)
// These are inline to avoid circular dependency during build
interface FalkorClientLike {
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  isConnected(): boolean;
  mergeNode(node: GraphNodeLike): Promise<void>;
  mergeEdge(edge: GraphEdgeLike): Promise<void>;
}

interface GraphNodeLike {
  id: string;
  labels: string[];
  properties: Record<string, unknown>;
}

interface GraphEdgeLike {
  type: string;
  from: string;
  to: string;
  properties?: Record<string, unknown>;
}

export interface GraphEmitConfig {
  /** FalkorDB client instance */
  client?: FalkorClientLike;
  /** Whether to emit to graph (can disable for testing) */
  enabled: boolean;
  /** Include LSA vectors in node properties */
  includeVectors: boolean;
  /** Create semantic similarity edges */
  createSimilarityEdges: boolean;
  /** Similarity threshold for edge creation */
  similarityThreshold: number;
  /** Auto-connect on first emit */
  autoConnect: boolean;
}

const DEFAULT_CONFIG: GraphEmitConfig = {
  enabled: true,
  includeVectors: true,
  createSimilarityEdges: true,
  similarityThreshold: 0.7,
  autoConnect: true,
};

export interface HolonGraphData {
  id: string;
  name: string;
  path: string;
  ring: 0 | 1 | 2 | 3;
  lsaVector?: number[];
  stabilityScore?: number;
  symbols?: string[];
  dependencies?: string[];
  language?: string;
  loc?: number;
}

export interface GraphEmitResult {
  success: boolean;
  nodeId?: string;
  edgesCreated?: number;
  error?: string;
}

/**
 * GraphEmitter: Handles emission of compiled artifacts to FalkorDB.
 */
export class GraphEmitter {
  private config: GraphEmitConfig;
  private client?: FalkorClientLike;
  private connected: boolean = false;

  constructor(config?: Partial<GraphEmitConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.client = config?.client;
  }

  /**
   * Set the FalkorDB client.
   */
  setClient(client: FalkorClientLike): void {
    this.client = client;
    this.connected = false;
  }

  /**
   * Ensure connection to FalkorDB.
   */
  async ensureConnected(): Promise<boolean> {
    if (!this.config.enabled || !this.client) {
      return false;
    }

    if (this.connected && this.client.isConnected()) {
      return true;
    }

    if (this.config.autoConnect) {
      try {
        await this.client.connect();
        this.connected = true;
        return true;
      } catch (error) {
        console.error('[GraphEmitter] Connection failed:', error);
        return false;
      }
    }

    return false;
  }

  /**
   * Emit a compiled artifact to the graph.
   */
  async emitArtifact(
    emitOutput: EmitOutput,
    holonData: HolonGraphData,
    context?: CompilationContext
  ): Promise<GraphEmitResult> {
    if (!this.config.enabled) {
      return { success: true, nodeId: holonData.id, edgesCreated: 0 };
    }

    const connected = await this.ensureConnected();
    if (!connected || !this.client) {
      return {
        success: false,
        error: 'Not connected to FalkorDB',
      };
    }

    try {
      // 1. Create/merge the holon node
      await this.mergeHolonNode(holonData, emitOutput);

      // 2. Create provenance lineage edges
      await this.createProvenanceEdges(holonData.id, emitOutput);

      // 3. Create dependency edges
      let edgesCreated = 0;
      if (holonData.dependencies) {
        for (const dep of holonData.dependencies) {
          await this.client.mergeEdge({
            type: 'DEPENDS_ON',
            from: holonData.id,
            to: dep,
          });
          edgesCreated++;
        }
      }

      // 4. Create symbol definition edges
      if (holonData.symbols) {
        for (const symbol of holonData.symbols) {
          await this.client.mergeEdge({
            type: 'DEFINES',
            from: holonData.id,
            to: `symbol:${symbol}`,
          });
          edgesCreated++;
        }
      }

      return {
        success: true,
        nodeId: holonData.id,
        edgesCreated,
      };
    } catch (error) {
      return {
        success: false,
        error: String(error),
      };
    }
  }

  /**
   * Merge a holon node into the graph.
   */
  private async mergeHolonNode(
    holon: HolonGraphData,
    emitOutput: EmitOutput
  ): Promise<void> {
    if (!this.client) return;

    const properties: Record<string, unknown> = {
      name: holon.name,
      path: holon.path,
      ring: holon.ring,
      language: holon.language,
      loc: holon.loc,
      stability_score: holon.stabilityScore ?? 0,
      trust_level: emitOutput.artifact.trust,
      compiled_at: emitOutput.artifact.compiledAt,
      verification_passed: emitOutput.ir.provenance?.entries.every(
        e => !e.taint || e.taint.length === 0
      ) ?? true,
    };

    if (this.config.includeVectors && holon.lsaVector) {
      properties.lsa_vector = holon.lsaVector;
    }

    await this.client.mergeNode({
      id: holon.id,
      labels: ['holon', `ring${holon.ring}`],
      properties,
    });
  }

  /**
   * Create provenance lineage edges from compilation history.
   */
  private async createProvenanceEdges(
    holonId: string,
    emitOutput: EmitOutput
  ): Promise<void> {
    if (!this.client || !emitOutput.ir.provenance) return;

    const entries = emitOutput.ir.provenance.entries;

    // Create lineage node for this compilation
    const lineageId = `lineage:${holonId}:${emitOutput.artifact.compiledAt}`;
    await this.client.mergeNode({
      id: lineageId,
      labels: ['lineage'],
      properties: {
        holon_id: holonId,
        timestamp: emitOutput.artifact.compiledAt,
        stages: entries.map(e => e.stage).join(','),
        trust: emitOutput.artifact.trust,
      },
    });

    // Link holon to lineage
    await this.client.mergeEdge({
      type: 'HAS_LINEAGE',
      from: holonId,
      to: lineageId,
    });
  }

  /**
   * Batch emit multiple holons with similarity edge creation.
   */
  async emitBatch(
    items: Array<{
      emitOutput: EmitOutput;
      holonData: HolonGraphData;
    }>
  ): Promise<{
    successful: number;
    failed: number;
    similarityEdgesCreated: number;
  }> {
    let successful = 0;
    let failed = 0;
    let similarityEdgesCreated = 0;

    // First pass: emit all nodes
    for (const item of items) {
      const result = await this.emitArtifact(item.emitOutput, item.holonData);
      if (result.success) {
        successful++;
      } else {
        failed++;
      }
    }

    // Second pass: create similarity edges if enabled
    if (this.config.createSimilarityEdges && this.client) {
      const holonsWithVectors = items.filter(i => i.holonData.lsaVector);

      for (let i = 0; i < holonsWithVectors.length; i++) {
        for (let j = i + 1; j < holonsWithVectors.length; j++) {
          const sim = this.cosineSimilarity(
            holonsWithVectors[i].holonData.lsaVector!,
            holonsWithVectors[j].holonData.lsaVector!
          );

          if (sim >= this.config.similarityThreshold) {
            await this.client.mergeEdge({
              type: 'SIMILAR_TO',
              from: holonsWithVectors[i].holonData.id,
              to: holonsWithVectors[j].holonData.id,
              properties: {
                similarity: sim,
                semantic_distance: 1 - sim,
              },
            });
            similarityEdgesCreated++;
          }
        }
      }
    }

    return { successful, failed, similarityEdgesCreated };
  }

  /**
   * Cosine similarity between two vectors.
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;

    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }
}

/**
 * Create a GraphEmitter with default configuration.
 */
export function createGraphEmitter(
  config?: Partial<GraphEmitConfig>
): GraphEmitter {
  return new GraphEmitter(config);
}

/**
 * Extend emit output with graph emission.
 * This is a drop-in enhancement for the emit stage.
 */
export async function emitToGraph(
  emitOutput: EmitOutput,
  holonData: HolonGraphData,
  emitter: GraphEmitter
): Promise<EmitOutput & { graphResult: GraphEmitResult }> {
  const graphResult = await emitter.emitArtifact(emitOutput, holonData);

  return {
    ...emitOutput,
    graphResult,
  };
}
