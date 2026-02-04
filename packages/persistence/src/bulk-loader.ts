/**
 * BulkLoader: High-speed CSV ingestion for FalkorDB
 *
 * Step 8 of NeSy Evolution: Generate CSVs for initial distillation ingestion.
 *
 * Workflow:
 * 1. Export code holons to CSV format
 * 2. Export relationships to CSV format
 * 3. Use FalkorDB bulk insert for performance
 *
 * CSV Format for Nodes:
 *   id,label,name,path,ring,lsa_vector,stability_score,...
 *
 * CSV Format for Edges:
 *   from_id,to_id,type,weight,semantic_distance,...
 */

import * as fs from 'fs';
import * as path from 'path';
import { LSAModel } from '@nesy/core';
import { FalkorClient, GraphNode, GraphEdge } from './falkor-client';

export interface HolonData {
  id: string;
  name: string;
  path: string;
  content: string;
  ring: number;
  stabilityScore?: number;
  metadata?: Record<string, unknown>;
}

export interface RelationshipData {
  from: string;
  to: string;
  type: 'IMPORTS' | 'CALLS' | 'EXTENDS' | 'SIMILAR_TO' | 'DEPENDS_ON';
  weight?: number;
  semanticDistance?: number;
}

export interface BulkLoaderConfig {
  outputDir: string;
  batchSize: number;
  includeVectors: boolean;
  vectorPrecision: number;
}

const DEFAULT_CONFIG: BulkLoaderConfig = {
  outputDir: './bulk-output',
  batchSize: 1000,
  includeVectors: true,
  vectorPrecision: 6
};

export class BulkLoader {
  private config: BulkLoaderConfig;
  private client: FalkorClient;

  constructor(client: FalkorClient, config?: Partial<BulkLoaderConfig>) {
    this.client = client;
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Export holons to CSV for bulk loading.
   */
  async exportHolonsToCSV(
    holons: HolonData[],
    lsaModel?: LSAModel,
    filename: string = 'holons.csv'
  ): Promise<string> {
    await fs.promises.mkdir(this.config.outputDir, { recursive: true });
    const filepath = path.join(this.config.outputDir, filename);

    const headers = ['id', 'name', 'path', 'ring', 'stability_score'];
    if (this.config.includeVectors && lsaModel) {
      headers.push('lsa_vector');
    }

    const rows: string[] = [headers.join(',')];

    for (const holon of holons) {
      const row: string[] = [
        this.escapeCSV(holon.id),
        this.escapeCSV(holon.name),
        this.escapeCSV(holon.path),
        String(holon.ring),
        String(holon.stabilityScore ?? 0)
      ];

      if (this.config.includeVectors && lsaModel) {
        const vector = this.getHolonVector(holon, lsaModel);
        row.push(this.serializeVector(vector));
      }

      rows.push(row.join(','));
    }

    await fs.promises.writeFile(filepath, rows.join('\n'), 'utf-8');
    console.log(`[BulkLoader] Exported ${holons.length} holons to ${filepath}`);

    return filepath;
  }

  /**
   * Export relationships to CSV for bulk loading.
   */
  async exportRelationshipsToCSV(
    relationships: RelationshipData[],
    filename: string = 'relationships.csv'
  ): Promise<string> {
    await fs.promises.mkdir(this.config.outputDir, { recursive: true });
    const filepath = path.join(this.config.outputDir, filename);

    const headers = ['from_id', 'to_id', 'type', 'weight', 'semantic_distance'];
    const rows: string[] = [headers.join(',')];

    for (const rel of relationships) {
      rows.push([
        this.escapeCSV(rel.from),
        this.escapeCSV(rel.to),
        rel.type,
        String(rel.weight ?? 1),
        String(rel.semanticDistance ?? 0)
      ].join(','));
    }

    await fs.promises.writeFile(filepath, rows.join('\n'), 'utf-8');
    console.log(`[BulkLoader] Exported ${relationships.length} relationships to ${filepath}`);

    return filepath;
  }

  /**
   * Generate semantic similarity edges from LSA model.
   */
  generateSimilarityEdges(
    lsaModel: LSAModel,
    threshold: number = 0.7
  ): RelationshipData[] {
    const edges: RelationshipData[] = [];
    const docs = lsaModel.tfidf.documents;
    const vectors = lsaModel.documentVectors;

    for (let i = 0; i < docs.length; i++) {
      for (let j = i + 1; j < docs.length; j++) {
        const similarity = this.cosineSimilarity(vectors[i], vectors[j]);
        if (similarity >= threshold) {
          edges.push({
            from: docs[i],
            to: docs[j],
            type: 'SIMILAR_TO',
            weight: similarity,
            semanticDistance: 1 - similarity
          });
        }
      }
    }

    console.log(`[BulkLoader] Generated ${edges.length} similarity edges (threshold: ${threshold})`);
    return edges;
  }

  /**
   * Perform bulk load from CSV files.
   */
  async bulkLoad(
    holonsCsv: string,
    relationshipsCsv: string,
    labels: string[] = ['Holon', 'Code']
  ): Promise<{ nodes: number; edges: number }> {
    const nodesLoaded = await this.client.bulkLoadNodes(holonsCsv, labels);
    const edgesLoaded = await this.client.bulkLoadEdges(relationshipsCsv, 'RELATES');

    return { nodes: nodesLoaded, edges: edgesLoaded };
  }

  /**
   * Full pipeline: Export and load holons with relationships.
   */
  async loadHolons(
    holons: HolonData[],
    lsaModel?: LSAModel,
    options: {
      generateSimilarityEdges?: boolean;
      similarityThreshold?: number;
      additionalRelationships?: RelationshipData[];
    } = {}
  ): Promise<{ nodes: number; edges: number }> {
    // Export holons
    const holonsCsv = await this.exportHolonsToCSV(holons, lsaModel);

    // Generate relationships
    const relationships: RelationshipData[] = [
      ...(options.additionalRelationships || [])
    ];

    if (options.generateSimilarityEdges && lsaModel) {
      const simEdges = this.generateSimilarityEdges(
        lsaModel,
        options.similarityThreshold ?? 0.7
      );
      relationships.push(...simEdges);
    }

    // Export relationships
    const relsCsv = await this.exportRelationshipsToCSV(relationships);

    // Perform bulk load
    return this.bulkLoad(holonsCsv, relsCsv);
  }

  /**
   * Get LSA vector for a holon by matching its path to the model.
   */
  private getHolonVector(holon: HolonData, lsaModel: LSAModel): number[] {
    const idx = lsaModel.tfidf.documents.indexOf(holon.path);
    if (idx >= 0 && idx < lsaModel.documentVectors.length) {
      return lsaModel.documentVectors[idx];
    }
    return [];
  }

  /**
   * Serialize a vector for CSV storage.
   */
  private serializeVector(vector: number[]): string {
    if (vector.length === 0) return '[]';
    const rounded = vector.map(v =>
      v.toFixed(this.config.vectorPrecision)
    );
    return `"[${rounded.join(',')}]"`;
  }

  /**
   * Escape a string for CSV.
   */
  private escapeCSV(value: string): string {
    if (value.includes(',') || value.includes('"') || value.includes('\n')) {
      return `"${value.replace(/"/g, '""')}"`;
    }
    return value;
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
