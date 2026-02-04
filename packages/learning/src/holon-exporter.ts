/**
 * HolonExporter - Export Gold Standard holons for training data generation
 *
 * Exports Ring 0/1 holons from FalkorDB into JSONL training pairs (Context -> Code)
 */

export type RingLevel = 0 | 1 | 2 | 3;

export interface TrainingPair {
  context: string;
  code: string;
  metadata: {
    holonId: string;
    ring: RingLevel;
    confidence: number;
    path: string;
    symbols?: string[];
    semanticCluster?: string;
  };
}

export interface ExportConfig {
  ringLevels: number[];
  minConfidence: number;
  outputPath: string;
}

export interface Holon {
  id: string;
  name: string;
  path: string;
  ring: RingLevel;
  content?: string;
  confidence?: number;
  symbols?: string[];
  dependencies?: string[];
  lsa_vector?: number[];
  semantic_cluster?: string;
  stability_score?: number;
  language?: string;
  loc?: number;
  last_modified?: string;
}

export class HolonExporter {
  private readonly falkordbPort: number;

  constructor(falkordbPort: number = 6379) {
    this.falkordbPort = falkordbPort;
  }

  async queryGoldHolons(config: ExportConfig): Promise<Holon[]> {
    const ringFilter = config.ringLevels.join(',');
    const query = `
      MATCH (h:Holon)
      WHERE h.ring IN [${ringFilter}]
        AND coalesce(h.stability_score, 1.0) >= ${config.minConfidence}
      RETURN h
      ORDER BY h.stability_score DESC
    `;

    // In production, this would connect to FalkorDB
    // For now, return empty array - integration tests will mock this
    console.log(`[HolonExporter] Query FalkorDB on port ${this.falkordbPort}: ${query.trim()}`);
    return [];
  }

  formatAsTrainingPair(holon: Holon): TrainingPair {
    const context = this.buildContext(holon);
    const code = holon.content ?? '';

    return {
      context,
      code,
      metadata: {
        holonId: holon.id,
        ring: holon.ring,
        confidence: holon.confidence ?? holon.stability_score ?? 1.0,
        path: holon.path,
        symbols: holon.symbols,
        semanticCluster: holon.semantic_cluster,
      },
    };
  }

  async exportToJSONL(holons: Holon[], outputPath: string): Promise<number> {
    const fs = await import('node:fs/promises');
    const path = await import('node:path');

    // Ensure directory exists
    const dir = path.dirname(outputPath);
    await fs.mkdir(dir, { recursive: true });

    const lines: string[] = [];
    for (const holon of holons) {
      const pair = this.formatAsTrainingPair(holon);
      lines.push(JSON.stringify(pair));
    }

    await fs.writeFile(outputPath, lines.join('\n') + (lines.length > 0 ? '\n' : ''));
    return lines.length;
  }

  private buildContext(holon: Holon): string {
    const parts: string[] = [];

    parts.push(`File: ${holon.path}`);
    if (holon.name) {
      parts.push(`Name: ${holon.name}`);
    }
    if (holon.language) {
      parts.push(`Language: ${holon.language}`);
    }
    if (holon.symbols && holon.symbols.length > 0) {
      parts.push(`Symbols: ${holon.symbols.join(', ')}`);
    }
    if (holon.dependencies && holon.dependencies.length > 0) {
      parts.push(`Dependencies: ${holon.dependencies.join(', ')}`);
    }
    if (holon.semantic_cluster) {
      parts.push(`Semantic Cluster: ${holon.semantic_cluster}`);
    }

    return parts.join('\n');
  }
}
