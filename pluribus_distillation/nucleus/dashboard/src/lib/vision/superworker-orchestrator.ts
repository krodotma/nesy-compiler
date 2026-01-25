/**
 * SUPERWORKER Orchestrator - Master Coordination for Naive→Agentic Bridge
 *
 * This is the central nervous system of the SUPERWORKERS architecture.
 * It coordinates:
 *   1. Vision capture and VLM analysis
 *   2. Gestalt comprehension aggregation
 *   3. 5-layer context injection
 *   4. Golden-ratio quality gating
 *   5. Bus event emission for observability
 *
 * The orchestrator transforms "naive" web chat workers into context-aware
 * agents by injecting the same architectural layers that make CLI tools
 * like Claude Code, Codex CLI, and Gemini CLI so effective.
 *
 * @module vision/superworker-orchestrator
 */

import { PHI, FIBONACCI, calculateGoldenScore, type CapturedFrame } from './screen-capture';
import type { WebRTCMetrics } from './webrtc-stats';
import type { VLMProviderName, VisionTaskType as VisionTask } from './vlm-integration';

// =============================================================================
// GOLDEN CONSTANTS - Extended Set
// =============================================================================

/** Euler's number for exponential decay */
const E = 2.718281828459045;

/** Silver ratio (δ_S = 1 + √2) for secondary optimization */
const SILVER_RATIO = 2.414213562373095;

/** Plastic constant (ρ ≈ 1.3247) for ternary scaling */
const PLASTIC_CONSTANT = 1.324717957244746;

/** Lucas numbers (related to Fibonacci, used for backup scaling) */
const LUCAS = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322, 521, 843] as const;

/** Token budgets using Fibonacci scaling */
const TOKEN_BUDGETS = {
  micro: FIBONACCI[4] * 100,      // 300 tokens
  mini: FIBONACCI[5] * 100,       // 500 tokens
  lite: FIBONACCI[7] * 100,       // 1,300 tokens
  standard: FIBONACCI[9] * 100,   // 3,400 tokens
  full: FIBONACCI[11] * 100,      // 8,900 tokens
  deep: FIBONACCI[13] * 100,      // 23,300 tokens
  ultra: FIBONACCI[15] * 100,     // 61,000 tokens
  maximum: FIBONACCI[17] * 100,   // 158,400 tokens
} as const;

/** Attention window sizes (φ-scaled from base 512) */
const ATTENTION_WINDOWS = {
  narrow: Math.round(512 / PHI),           // 316 tokens
  focused: 512,                             // 512 tokens
  balanced: Math.round(512 * PHI),         // 828 tokens
  wide: Math.round(512 * PHI * PHI),       // 1,340 tokens
  panoramic: Math.round(512 * PHI ** 3),   // 2,167 tokens
  omniscient: Math.round(512 * PHI ** 4),  // 3,505 tokens
} as const;

// =============================================================================
// TYPES - Comprehensive Type System
// =============================================================================

/**
 * The 5 injection layers that transform naive chat to agentic operation.
 * Each layer adds progressively more contextual richness.
 */
export type InjectionLayer =
  | 'constitutional'  // Safety rails, ethical constraints, forbidden actions
  | 'idiolect'        // Domain-specific grammar, terminology, patterns
  | 'persona'         // Role expertise, behavioral traits, response style
  | 'context'         // Codebase state, file contents, dependency graph
  | 'observable';     // Live tools, environment state, active processes

/**
 * Depth classification for routing queries to appropriate processing tiers.
 */
export type DepthLevel = 'micro' | 'narrow' | 'standard' | 'deep' | 'omniscient';

/**
 * Effects budget classification (P/E/L/R/Q gates).
 */
export type EffectsBudget = 'none' | 'read' | 'write' | 'network' | 'execute' | 'unknown';

/**
 * Coordination lanes for multi-agent communication.
 */
export type CoordinationLane = 'dialogos' | 'pbpair' | 'strp';

/**
 * Quality gates derived from golden ratio.
 */
export interface QualityGates {
  /** Provenance: Source trustworthiness (0-1) */
  provenance: number;
  /** Effects: Side-effect risk level (0-1, lower is safer) */
  effects: number;
  /** Liveness: Real-time validity (0-1) */
  liveness: number;
  /** Recovery: Rollback capability (0-1) */
  recovery: number;
  /** Quality: Overall output quality (0-PHI) */
  quality: number;
}

/**
 * Gestalt comprehension state - holistic codebase understanding.
 */
export interface GestaltState {
  /** High-level architectural summary */
  architectureSummary: string;
  /** Key files and their purposes */
  keyFiles: Map<string, string>;
  /** Dependency graph (simplified) */
  dependencies: Map<string, string[]>;
  /** Active concerns/contexts */
  activeConcerns: string[];
  /** Confidence in understanding (0-1) */
  confidence: number;
  /** Last update timestamp */
  updatedAt: number;
  /** Golden comprehension score */
  goldenScore: number;
}

/**
 * Injected context ready for message augmentation.
 */
export interface InjectedContext {
  /** The injection layer this context belongs to */
  layer: InjectionLayer;
  /** Priority weight (φ-scaled) */
  weight: number;
  /** Token count for this context chunk */
  tokenCount: number;
  /** The actual content to inject */
  content: string;
  /** Metadata for debugging/observability */
  metadata: Record<string, unknown>;
}

/**
 * Complete SUPERWORKER state for a session.
 */
export interface SuperworkerState {
  /** Unique session identifier */
  sessionId: string;
  /** Current depth classification */
  depthLevel: DepthLevel;
  /** Active coordination lanes */
  lanes: CoordinationLane[];
  /** Effects budget for this session */
  effectsBudget: EffectsBudget[];
  /** Quality gates state */
  qualityGates: QualityGates;
  /** Gestalt comprehension */
  gestalt: GestaltState;
  /** Vision context (if available) */
  visionContext?: {
    frame: CapturedFrame;
    metrics?: WebRTCMetrics;
    provider?: VLMProviderName;
  };
  /** All injected contexts by layer */
  injectedContexts: Map<InjectionLayer, InjectedContext[]>;
  /** Total token budget consumed */
  tokensConsumed: number;
  /** Maximum token budget */
  tokenBudget: number;
  /** Golden optimization score */
  goldenScore: number;
  /** Creation timestamp */
  createdAt: number;
  /** Last activity timestamp */
  lastActivityAt: number;
}

/**
 * Orchestrator configuration options.
 */
export interface OrchestratorConfig {
  /** Base token budget tier */
  budgetTier: keyof typeof TOKEN_BUDGETS;
  /** Attention window size */
  attentionWindow: keyof typeof ATTENTION_WINDOWS;
  /** Enabled injection layers */
  enabledLayers: InjectionLayer[];
  /** Minimum golden score for operations */
  minGoldenScore: number;
  /** Whether to emit bus events */
  emitBusEvents: boolean;
  /** Vision capture options */
  visionOptions?: {
    enabled: boolean;
    provider?: VLMProviderName;
    task?: VisionTask;
    captureInterval?: number;
  };
  /** Persona ID from personas.json */
  personaId?: string;
}

/**
 * Query analysis result for depth classification.
 */
export interface QueryAnalysis {
  /** Classified depth level */
  depth: DepthLevel;
  /** Confidence in classification (0-1) */
  confidence: number;
  /** Detected intents */
  intents: string[];
  /** Extracted entities (files, functions, concepts) */
  entities: string[];
  /** Suggested coordination lanes */
  suggestedLanes: CoordinationLane[];
  /** Estimated token requirement */
  estimatedTokens: number;
  /** Golden score for query quality */
  goldenScore: number;
}

// =============================================================================
// DEPTH CLASSIFIER - NLP-based Query Analysis
// =============================================================================

/**
 * Pattern sets for depth classification.
 * Uses linguistic markers to identify query complexity.
 */
const DEPTH_PATTERNS = {
  micro: {
    keywords: [
      'what is', 'define', 'explain', 'meaning of', 'tell me about',
      'quick', 'simple', 'brief', 'short', 'summary'
    ],
    maxLength: 50,
    maxEntities: 1,
    weight: 1 / (PHI ** 3),  // ~0.236
  },
  narrow: {
    keywords: [
      'how do i', 'where is', 'find', 'locate', 'show me',
      'fix', 'change', 'update', 'modify', 'edit',
      'single', 'one', 'specific', 'particular'
    ],
    maxLength: 150,
    maxEntities: 3,
    weight: 1 / (PHI ** 2),  // ~0.382
  },
  standard: {
    keywords: [
      'implement', 'create', 'build', 'add feature', 'write',
      'refactor', 'improve', 'optimize', 'debug', 'test',
      'several', 'multiple', 'various', 'different'
    ],
    maxLength: 400,
    maxEntities: 7,
    weight: 1 / PHI,  // ~0.618
  },
  deep: {
    keywords: [
      'architecture', 'design', 'system', 'integrate', 'migrate',
      'comprehensive', 'complete', 'full', 'entire', 'all',
      'analyze', 'investigate', 'research', 'understand',
      'pattern', 'strategy', 'approach'
    ],
    maxLength: 1000,
    maxEntities: 15,
    weight: 1.0,
  },
  omniscient: {
    keywords: [
      'everything', 'entire codebase', 'whole project', 'all files',
      'holistic', 'gestalt', 'big picture', 'overview',
      'restructure', 'rewrite', 'overhaul', 'transform',
      'cross-cutting', 'pervasive', 'fundamental'
    ],
    maxLength: Infinity,
    maxEntities: Infinity,
    weight: PHI,  // ~1.618
  },
} as const;

/**
 * Intent patterns for understanding query purpose.
 */
const INTENT_PATTERNS = {
  read: /\b(show|display|get|fetch|read|view|list|find|search|locate|where)\b/i,
  write: /\b(create|write|add|insert|new|generate|make|build)\b/i,
  modify: /\b(change|update|modify|edit|fix|patch|adjust|alter|refactor)\b/i,
  delete: /\b(remove|delete|drop|clear|clean|purge|eliminate)\b/i,
  analyze: /\b(analyze|examine|investigate|debug|understand|explain|why|how)\b/i,
  execute: /\b(run|execute|start|launch|deploy|trigger|invoke|call)\b/i,
  plan: /\b(plan|design|architect|strategy|approach|consider|think)\b/i,
} as const;

/**
 * Entity extraction patterns.
 */
const ENTITY_PATTERNS = {
  file: /\b[\w-]+\.(ts|tsx|js|jsx|py|md|json|yaml|yml|toml|sh|css|scss)\b/gi,
  function: /\b(function|func|def|const|let|var|class)\s+(\w+)/gi,
  path: /\/?[\w-]+(?:\/[\w-]+)+(?:\.\w+)?/g,
  component: /<(\w+)[\s/>]/g,
  variable: /\b[A-Z][A-Z_0-9]+\b/g,  // CONSTANTS
  camelCase: /\b[a-z]+(?:[A-Z][a-z]+)+\b/g,  // camelCase identifiers
};

/**
 * Analyze a query to determine depth, intents, and entities.
 * Uses golden-ratio weighted scoring for classification confidence.
 */
export function analyzeQuery(query: string): QueryAnalysis {
  const normalizedQuery = query.toLowerCase().trim();
  const queryLength = query.length;

  // Extract entities
  const entities: string[] = [];
  for (const [_type, pattern] of Object.entries(ENTITY_PATTERNS)) {
    const matches = query.match(pattern) || [];
    entities.push(...matches);
  }
  const uniqueEntities = [...new Set(entities)];

  // Detect intents
  const intents: string[] = [];
  for (const [intent, pattern] of Object.entries(INTENT_PATTERNS)) {
    if (pattern.test(normalizedQuery)) {
      intents.push(intent);
    }
  }

  // Score each depth level
  const depthScores: Record<DepthLevel, number> = {
    micro: 0,
    narrow: 0,
    standard: 0,
    deep: 0,
    omniscient: 0,
  };

  for (const [level, config] of Object.entries(DEPTH_PATTERNS)) {
    const depth = level as DepthLevel;
    let score = 0;

    // Keyword matching (φ-weighted)
    const keywordMatches = config.keywords.filter(kw =>
      normalizedQuery.includes(kw)
    ).length;
    score += keywordMatches * config.weight * PHI;

    // Length heuristic
    if (queryLength <= config.maxLength) {
      score += config.weight;
    }

    // Entity count heuristic
    if (uniqueEntities.length <= config.maxEntities) {
      score += config.weight / PHI;
    }

    depthScores[depth] = score;
  }

  // Find best depth match
  let bestDepth: DepthLevel = 'standard';
  let bestScore = 0;
  for (const [depth, score] of Object.entries(depthScores)) {
    if (score > bestScore) {
      bestScore = score;
      bestDepth = depth as DepthLevel;
    }
  }

  // Calculate confidence using sigmoid normalization
  const totalScore = Object.values(depthScores).reduce((a, b) => a + b, 0);
  const confidence = totalScore > 0
    ? bestScore / totalScore
    : 1 / (PHI * PHI);  // Default fair confidence

  // Suggest coordination lanes based on depth and intents
  const suggestedLanes: CoordinationLane[] = [];
  if (bestDepth === 'micro' || bestDepth === 'narrow') {
    suggestedLanes.push('dialogos');
  }
  if (bestDepth === 'standard' || bestDepth === 'deep') {
    suggestedLanes.push('pbpair');
  }
  if (bestDepth === 'deep' || bestDepth === 'omniscient') {
    suggestedLanes.push('strp');
  }
  if (intents.includes('execute') || intents.includes('write')) {
    if (!suggestedLanes.includes('pbpair')) {
      suggestedLanes.push('pbpair');
    }
  }

  // Estimate token requirement using Fibonacci scaling
  const baseTokens = TOKEN_BUDGETS[
    bestDepth === 'micro' ? 'micro' :
    bestDepth === 'narrow' ? 'lite' :
    bestDepth === 'standard' ? 'standard' :
    bestDepth === 'deep' ? 'deep' : 'ultra'
  ];
  const entityMultiplier = 1 + (uniqueEntities.length * 0.1);
  const intentMultiplier = 1 + (intents.length * 0.05);
  const estimatedTokens = Math.round(baseTokens * entityMultiplier * intentMultiplier);

  // Calculate golden score for query quality
  const goldenScore = calculateGoldenScore({
    resolution: confidence,
    frameRate: Math.min(intents.length / 3, 1),
    stability: Math.min(uniqueEntities.length / 10, 1),
    latency: queryLength > 20 ? 0.8 : 0.5,
  });

  return {
    depth: bestDepth,
    confidence,
    intents,
    entities: uniqueEntities,
    suggestedLanes,
    estimatedTokens,
    goldenScore,
  };
}

// =============================================================================
// GESTALT COMPREHENSION - Holistic Understanding Aggregator
// =============================================================================

/**
 * Context source for gestalt aggregation.
 */
export interface ContextSource {
  type: 'file' | 'symbol' | 'dependency' | 'history' | 'vision' | 'bus';
  path?: string;
  content: string;
  relevance: number;  // 0-1, how relevant to current query
  freshness: number;  // 0-1, how recent (1 = just captured)
  confidence: number; // 0-1, confidence in accuracy
}

/**
 * Aggregate multiple context sources into gestalt understanding.
 * Uses φ-weighted importance scoring for prioritization.
 */
export function aggregateGestalt(
  sources: ContextSource[],
  maxTokens: number
): GestaltState {
  // Score each source using golden ratio weighting
  const scoredSources = sources.map(source => {
    const typeWeight = {
      vision: PHI,          // Most immediate
      file: 1.0,            // Primary source
      symbol: 1 / PHI,      // Derived from files
      history: 1 / (PHI * PHI),  // May be stale
      dependency: 1 / (PHI * PHI),
      bus: 1 / (PHI ** 3),  // Auxiliary
    }[source.type];

    // Geometric mean of factors with φ-weighting
    const score = Math.pow(
      Math.pow(source.relevance, PHI) *
      Math.pow(source.freshness, 1.0) *
      Math.pow(source.confidence, 1 / PHI) *
      typeWeight,
      1 / 3
    );

    return { source, score };
  });

  // Sort by score descending
  scoredSources.sort((a, b) => b.score - a.score);

  // Aggregate until token budget exhausted
  const keyFiles = new Map<string, string>();
  const dependencies = new Map<string, string[]>();
  const activeConcerns: string[] = [];
  let architectureSummary = '';
  let totalTokens = 0;
  let totalScore = 0;
  let sourceCount = 0;

  for (const { source, score } of scoredSources) {
    // Estimate tokens (rough: 4 chars per token)
    const estimatedTokens = Math.ceil(source.content.length / 4);

    if (totalTokens + estimatedTokens > maxTokens) {
      // Apply diminishing returns with silver ratio
      if (totalTokens + estimatedTokens / SILVER_RATIO > maxTokens) {
        break;
      }
    }

    // Incorporate source
    if (source.type === 'file' && source.path) {
      keyFiles.set(source.path, source.content.slice(0, 500));
    } else if (source.type === 'dependency' && source.path) {
      const deps = source.content.split(',').map(d => d.trim());
      dependencies.set(source.path, deps);
    } else if (source.type === 'vision') {
      activeConcerns.push(`[VISION] ${source.content.slice(0, 200)}`);
    }

    totalTokens += estimatedTokens;
    totalScore += score;
    sourceCount++;
  }

  // Generate architecture summary from top sources
  const topSources = scoredSources.slice(0, 5);
  architectureSummary = topSources
    .map(s => s.source.path || s.source.type)
    .join(' → ');

  // Calculate overall gestalt golden score
  const avgScore = sourceCount > 0 ? totalScore / sourceCount : 0;
  const coverageScore = Math.min(sourceCount / 10, 1);
  const goldenScore = calculateGoldenScore({
    resolution: avgScore,
    frameRate: coverageScore,
    stability: keyFiles.size > 0 ? 0.9 : 0.5,
  });

  return {
    architectureSummary,
    keyFiles,
    dependencies,
    activeConcerns,
    confidence: avgScore,
    updatedAt: Date.now(),
    goldenScore,
  };
}

// =============================================================================
// CONTEXT INJECTOR - 5-Layer Injection Engine
// =============================================================================

/**
 * Layer-specific injection templates.
 */
const LAYER_TEMPLATES: Record<InjectionLayer, {
  prefix: string;
  suffix: string;
  weight: number;
  maxTokens: number;
}> = {
  constitutional: {
    prefix: '<constitutional_layer>\n',
    suffix: '\n</constitutional_layer>',
    weight: PHI ** 2,  // Highest priority
    maxTokens: FIBONACCI[8] * 10,  // 340 tokens
  },
  idiolect: {
    prefix: '<idiolect_layer domain="pluribus">\n',
    suffix: '\n</idiolect_layer>',
    weight: PHI,
    maxTokens: FIBONACCI[10] * 10,  // 550 tokens
  },
  persona: {
    prefix: '<persona_layer>\n',
    suffix: '\n</persona_layer>',
    weight: 1.0,
    maxTokens: FIBONACCI[11] * 10,  // 890 tokens
  },
  context: {
    prefix: '<context_layer>\n',
    suffix: '\n</context_layer>',
    weight: 1 / PHI,
    maxTokens: FIBONACCI[13] * 10,  // 2,330 tokens
  },
  observable: {
    prefix: '<observable_layer>\n',
    suffix: '\n</observable_layer>',
    weight: 1 / (PHI * PHI),
    maxTokens: FIBONACCI[12] * 10,  // 1,440 tokens
  },
};

/**
 * Build injection content for a specific layer.
 */
export function buildLayerInjection(
  layer: InjectionLayer,
  contents: string[],
  metadata: Record<string, unknown> = {}
): InjectedContext {
  const template = LAYER_TEMPLATES[layer];
  const combined = contents.join('\n\n');
  const truncated = combined.slice(0, template.maxTokens * 4);  // ~4 chars/token

  const content = `${template.prefix}${truncated}${template.suffix}`;
  const tokenCount = Math.ceil(content.length / 4);

  return {
    layer,
    weight: template.weight,
    tokenCount,
    content,
    metadata: {
      ...metadata,
      originalLength: combined.length,
      truncated: combined.length > truncated.length,
      layerMaxTokens: template.maxTokens,
    },
  };
}

/**
 * Assemble all layers into final injection payload.
 * Respects token budget with φ-weighted prioritization.
 */
export function assembleInjection(
  contexts: InjectedContext[],
  maxTokens: number
): { content: string; tokensUsed: number; layersIncluded: InjectionLayer[] } {
  // Sort by weight (highest first)
  const sorted = [...contexts].sort((a, b) => b.weight - a.weight);

  const included: InjectedContext[] = [];
  let totalTokens = 0;

  for (const ctx of sorted) {
    if (totalTokens + ctx.tokenCount <= maxTokens) {
      included.push(ctx);
      totalTokens += ctx.tokenCount;
    } else {
      // Try to fit partial content
      const remainingTokens = maxTokens - totalTokens;
      if (remainingTokens > 100) {
        // At least 100 tokens to be useful
        const ratio = remainingTokens / ctx.tokenCount;
        const truncatedContent = ctx.content.slice(0, Math.floor(ctx.content.length * ratio));
        included.push({
          ...ctx,
          content: truncatedContent + '\n[TRUNCATED]',
          tokenCount: remainingTokens,
        });
        totalTokens = maxTokens;
      }
      break;
    }
  }

  // Reorder by layer hierarchy (constitutional first, observable last)
  const layerOrder: InjectionLayer[] = ['constitutional', 'idiolect', 'persona', 'context', 'observable'];
  included.sort((a, b) => layerOrder.indexOf(a.layer) - layerOrder.indexOf(b.layer));

  return {
    content: included.map(c => c.content).join('\n\n'),
    tokensUsed: totalTokens,
    layersIncluded: included.map(c => c.layer),
  };
}

// =============================================================================
// SUPERWORKER ORCHESTRATOR CLASS
// =============================================================================

/**
 * Master orchestrator for SUPERWORKER operations.
 * Coordinates all subsystems with golden-ratio optimization.
 */
export class SuperworkerOrchestrator {
  private config: OrchestratorConfig;
  private state: SuperworkerState;
  private busEmitter: ((event: BusEvent) => void) | null = null;

  constructor(config: Partial<OrchestratorConfig> = {}) {
    this.config = {
      budgetTier: config.budgetTier || 'standard',
      attentionWindow: config.attentionWindow || 'balanced',
      enabledLayers: config.enabledLayers || ['constitutional', 'persona', 'context'],
      minGoldenScore: config.minGoldenScore || 1 / (PHI * PHI),  // 0.382 (fair)
      emitBusEvents: config.emitBusEvents ?? true,
      visionOptions: config.visionOptions,
      personaId: config.personaId,
    };

    this.state = this.initializeState();
  }

  private initializeState(): SuperworkerState {
    return {
      sessionId: `sw_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      depthLevel: 'standard',
      lanes: ['dialogos'],
      effectsBudget: ['none', 'read'],
      qualityGates: {
        provenance: 1.0,
        effects: 0.0,
        liveness: 1.0,
        recovery: 1.0,
        quality: 1 / PHI,  // Start at "good"
      },
      gestalt: {
        architectureSummary: '',
        keyFiles: new Map(),
        dependencies: new Map(),
        activeConcerns: [],
        confidence: 0,
        updatedAt: Date.now(),
        goldenScore: 0,
      },
      injectedContexts: new Map(),
      tokensConsumed: 0,
      tokenBudget: TOKEN_BUDGETS[this.config.budgetTier],
      goldenScore: 1 / PHI,
      createdAt: Date.now(),
      lastActivityAt: Date.now(),
    };
  }

  /**
   * Process a query through the full SUPERWORKER pipeline.
   */
  async processQuery(query: string, additionalContext?: string[]): Promise<{
    augmentedQuery: string;
    analysis: QueryAnalysis;
    injectedLayers: InjectionLayer[];
    tokensUsed: number;
    goldenScore: number;
  }> {
    // Step 1: Analyze query depth and intents
    const analysis = analyzeQuery(query);
    this.state.depthLevel = analysis.depth;
    this.state.lanes = analysis.suggestedLanes;

    // Step 2: Check quality gate
    if (analysis.goldenScore < this.config.minGoldenScore) {
      this.emitEvent('quality.warning', {
        score: analysis.goldenScore,
        threshold: this.config.minGoldenScore,
        query: query.slice(0, 100),
      });
    }

    // Step 3: Build context sources
    const sources: ContextSource[] = [];

    // Add additional context as file sources
    if (additionalContext) {
      for (const ctx of additionalContext) {
        sources.push({
          type: 'file',
          content: ctx,
          relevance: 0.8,
          freshness: 1.0,
          confidence: 0.9,
        });
      }
    }

    // Add vision context if enabled
    if (this.state.visionContext?.frame) {
      sources.push({
        type: 'vision',
        content: `[Screen capture: ${this.state.visionContext.frame.width}x${this.state.visionContext.frame.height}, golden score: ${this.state.visionContext.frame.goldenScore.toFixed(3)}]`,
        relevance: 0.9,
        freshness: 1.0,
        confidence: this.state.visionContext.frame.goldenScore,
      });
    }

    // Step 4: Aggregate gestalt
    const gestaltBudget = Math.round(this.state.tokenBudget * (1 / PHI));
    this.state.gestalt = aggregateGestalt(sources, gestaltBudget);

    // Step 5: Build injection layers
    const injections: InjectedContext[] = [];

    for (const layer of this.config.enabledLayers) {
      const layerContent = this.buildLayerContent(layer, analysis, query);
      if (layerContent.length > 0) {
        injections.push(buildLayerInjection(layer, layerContent, {
          query: query.slice(0, 50),
          depth: analysis.depth,
        }));
      }
    }

    // Step 6: Assemble final injection
    const remainingBudget = this.state.tokenBudget - this.state.tokensConsumed;
    const assembled = assembleInjection(injections, remainingBudget);

    // Step 7: Augment query
    const augmentedQuery = assembled.content
      ? `${assembled.content}\n\n---\n\n${query}`
      : query;

    // Update state
    this.state.tokensConsumed += assembled.tokensUsed;
    this.state.lastActivityAt = Date.now();
    this.state.goldenScore = calculateGoldenScore({
      resolution: analysis.confidence,
      frameRate: assembled.layersIncluded.length / 5,
      stability: this.state.gestalt.goldenScore,
    });

    // Emit completion event
    this.emitEvent('query.processed', {
      sessionId: this.state.sessionId,
      depth: analysis.depth,
      layersIncluded: assembled.layersIncluded,
      tokensUsed: assembled.tokensUsed,
      goldenScore: this.state.goldenScore,
    });

    return {
      augmentedQuery,
      analysis,
      injectedLayers: assembled.layersIncluded,
      tokensUsed: assembled.tokensUsed,
      goldenScore: this.state.goldenScore,
    };
  }

  /**
   * Build content for a specific injection layer.
   */
  private buildLayerContent(
    layer: InjectionLayer,
    analysis: QueryAnalysis,
    query: string
  ): string[] {
    const contents: string[] = [];

    switch (layer) {
      case 'constitutional':
        contents.push(`You are operating as a SUPERWORKER with depth level: ${analysis.depth}`);
        contents.push(`Quality threshold: ${this.config.minGoldenScore.toFixed(3)} (golden ratio scaled)`);
        contents.push(`Effects budget: ${this.state.effectsBudget.join(', ')}`);
        if (analysis.intents.includes('execute') || analysis.intents.includes('write')) {
          contents.push('CAUTION: Write/execute intents detected. Confirm before side effects.');
        }
        break;

      case 'idiolect':
        contents.push('Domain: Pluribus multi-agent orchestration system');
        contents.push('Key terms: bus events (NDJSON), golden scoring (φ), Fibonacci budgets');
        contents.push('Patterns: P/E/L/R/Q gates, 3-lane coordination, 5-layer injection');
        break;

      case 'persona':
        if (this.config.personaId) {
          contents.push(`Active persona: ${this.config.personaId}`);
        }
        contents.push(`Coordination lanes: ${this.state.lanes.join(', ')}`);
        contents.push(`Suggested approach: ${analysis.depth === 'deep' ? 'comprehensive analysis' : 'focused response'}`);
        break;

      case 'context':
        if (this.state.gestalt.architectureSummary) {
          contents.push(`Architecture: ${this.state.gestalt.architectureSummary}`);
        }
        if (this.state.gestalt.keyFiles.size > 0) {
          contents.push(`Key files: ${[...this.state.gestalt.keyFiles.keys()].slice(0, 5).join(', ')}`);
        }
        if (this.state.gestalt.activeConcerns.length > 0) {
          contents.push(`Active concerns: ${this.state.gestalt.activeConcerns.join('; ')}`);
        }
        break;

      case 'observable':
        contents.push(`Session: ${this.state.sessionId}`);
        contents.push(`Tokens consumed: ${this.state.tokensConsumed}/${this.state.tokenBudget}`);
        contents.push(`Golden score: ${this.state.goldenScore.toFixed(3)}`);
        if (this.state.visionContext) {
          contents.push(`Vision active: ${this.state.visionContext.provider || 'pending'}`);
        }
        break;
    }

    return contents;
  }

  /**
   * Inject vision context from screen capture.
   */
  setVisionContext(frame: CapturedFrame, provider?: VLMProviderName, metrics?: WebRTCMetrics): void {
    this.state.visionContext = { frame, provider, metrics };
    this.state.lastActivityAt = Date.now();

    this.emitEvent('vision.injected', {
      sessionId: this.state.sessionId,
      frameSize: `${frame.width}x${frame.height}`,
      provider,
      goldenScore: frame.goldenScore,
    });
  }

  /**
   * Get current orchestrator state.
   */
  getState(): Readonly<SuperworkerState> {
    return this.state;
  }

  /**
   * Get remaining token budget.
   */
  getRemainingBudget(): number {
    return this.state.tokenBudget - this.state.tokensConsumed;
  }

  /**
   * Reset orchestrator for new session.
   */
  reset(): void {
    const oldSessionId = this.state.sessionId;
    this.state = this.initializeState();

    this.emitEvent('session.reset', {
      oldSessionId,
      newSessionId: this.state.sessionId,
    });
  }

  /**
   * Set bus event emitter.
   */
  setBusEmitter(emitter: (event: BusEvent) => void): void {
    this.busEmitter = emitter;
  }

  /**
   * Emit event to bus (if configured).
   */
  private emitEvent(topic: string, data: Record<string, unknown>): void {
    if (!this.config.emitBusEvents) return;

    const event: BusEvent = {
      timestamp: Date.now(),
      topic: `superworker.${topic}`,
      kind: 'metric',
      level: 'debug',
      data: {
        ...data,
        phi: PHI,
      },
    };

    if (this.busEmitter) {
      this.busEmitter(event);
    }

    // Also try window bus
    try {
      if (typeof window !== 'undefined' && (window as any).__PLURIBUS_BUS__) {
        (window as any).__PLURIBUS_BUS__.emit(event);
      }
    } catch {
      // Silently ignore
    }
  }
}

// =============================================================================
// BUS EVENT TYPE
// =============================================================================

interface BusEvent {
  timestamp: number;
  topic: string;
  kind: 'metric' | 'log' | 'event';
  level: 'debug' | 'info' | 'warn' | 'error';
  data: Record<string, unknown>;
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  TOKEN_BUDGETS,
  ATTENTION_WINDOWS,
  DEPTH_PATTERNS,
  LAYER_TEMPLATES,
  PHI,
  FIBONACCI,
  SILVER_RATIO,
  PLASTIC_CONSTANT,
  LUCAS,
};

export default SuperworkerOrchestrator;
