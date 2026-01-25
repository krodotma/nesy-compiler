/**
 * Holon Schema Definitions
 * [Ultrathink Agent 1: Architect]
 * 
 * Defines the structural contract for the Holon Lineage system.
 * This schema bridges the gap between raw Git history and the "Holon" concept.
 */

export interface HolonNode {
  id: string;           // Git hash or component ID
  name: string;         // Component name (e.g. "IPEPanel")
  type: 'root' | 'trunk' | 'branch' | 'leaf';
  
  // Metrics
  cmp: number;          // Clade-Metaproductivity Score (0-100)
  complexity: number;   // Visualized as vertex count (LOC proxy)
  stability: number;    // Visualized as rotation smoothness
  
  // Lineage
  generation: number;
  parents: string[];
  children: string[];
  
  // Ontological Metadata
  etymon: string;       // The "True Name" / Semantic Hash
  tags: string[];
}

export interface HolonContext {
  activeNode: HolonNode;
  lineage: HolonNode[]; // Local neighborhood for visualization
}
