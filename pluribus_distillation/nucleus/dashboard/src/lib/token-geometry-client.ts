/**
 * Token Geometry Client - TypeScript interface for Token Geometry Engine
 *
 * Provides browser-side access to:
 * - Token counting and truncation
 * - AUOM (Atomic Units of Meaning) analysis
 * - Sextet vector encoding
 * - N-sphere projections
 * - Superpositional LTL validation
 * - Vec2Vec transformations
 *
 * Integration with dashboard components:
 * - NotificationSidepanel: Event clustering via n-sphere geometry
 * - WebLLMWidget: Token budget management
 * - DialogosContainer: Conversation state validation via LTL
 * - BusPulseWidget: Superpositional state visualization
 */

// =============================================================================
// TYPES
// =============================================================================

/** Sextet channel identifiers */
export type SextetChannel =
  | 'semantic'
  | 'syntactic'
  | 'pragmatic'
  | 'temporal'
  | 'modal'
  | 'affective';

/** AUOM category identifiers */
export type AUOMCategory =
  | 'entity'
  | 'action'
  | 'relation'
  | 'modifier'
  | 'quantifier'
  | 'temporal'
  | 'spatial'
  | 'operator'
  | 'connector';

/** LTL operators */
export type LTLOperator =
  | 'always' // □
  | 'eventually' // ◇
  | 'next' // ○
  | 'until' // U
  | 'release' // R
  | 'and' // ∧
  | 'or' // ∨
  | 'not' // ¬
  | 'implies'; // →

/** BPE Token */
export interface BPEToken {
  id: number;
  text: string;
  position: number;
  byteOffset: number;
}

/** Atomic Unit of Meaning */
export interface AUOMUnit {
  id: string;
  text: string;
  category: AUOMCategory;
  confidence: number;
  span: [number, number];
  embedding?: number[];
  metadata?: Record<string, unknown>;
}

/** Sextet vector (6-channel representation) */
export interface SextetVector {
  semantic: number[];
  syntactic: number[];
  pragmatic: number[];
  temporal: number[];
  modal: number[];
  affective: number[];
}

/** Point on n-dimensional hypersphere */
export interface NSpherePoint {
  coords: number[];
  radius: number;
  angular?: number[];
}

/** Superpositional state */
export interface SuperpositionalState {
  basisStates: string[];
  amplitudes: { real: number; imag: number }[];
}

/** LTL Formula */
export interface LTLFormula {
  operator: LTLOperator | null;
  operands: (LTLFormula | string)[];
}

/** Token geometry processing result */
export interface TokenGeometryResult {
  rawText: string;
  bpeTokens: BPEToken[];
  tokenCount: number;
  auomUnits: AUOMUnit[];
  detectedOperators: string[];
  sextetVector: SextetVector | null;
  nspherePoint: NSpherePoint | null;
  ltlFormula: LTLFormula | null;
  ltlValid: boolean;
  superposition: SuperpositionalState | null;
  processingTimeMs: number;
  engineVersion: string;
}

/** Vec2Vec transform specification */
export interface Vec2VecTransform {
  name: string;
  sourceSpace: string;
  targetSpace: string;
}

// =============================================================================
// BROWSER-SIDE IMPLEMENTATIONS
// =============================================================================

/**
 * Simple hash-based embedding for browser use
 * Uses Web Crypto API for deterministic hashing
 */
async function hashEmbed(text: string, dim: number = 64): Promise<number[]> {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);

  // Use SHA-256 for initial hash
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = new Uint8Array(hashBuffer);

  // Expand to desired dimension using SHAKE-like expansion
  const result: number[] = [];
  let seed = 0;

  for (let i = 0; i < dim; i++) {
    const idx = i % hashArray.length;
    // Pseudo-random expansion using hash bytes
    seed = (seed * 1103515245 + hashArray[idx] + 12345) >>> 0;
    result.push((seed / 0xffffffff) * 2 - 1); // Normalize to [-1, 1]
  }

  // Normalize to unit vector
  const norm = Math.sqrt(result.reduce((sum, v) => sum + v * v, 0));
  return norm > 1e-10 ? result.map((v) => v / norm) : result;
}

/**
 * Simple BPE-like tokenization for browser
 * Approximates tiktoken cl100k_base behavior
 */
function tokenizeSimple(text: string): BPEToken[] {
  const tokens: BPEToken[] = [];
  // Split on whitespace and punctuation boundaries
  const pattern = /(\s+|[.,!?;:'"()\[\]{}])|(\S+)/g;
  let match: RegExpExecArray | null;
  let position = 0;
  let byteOffset = 0;

  while ((match = pattern.exec(text)) !== null) {
    const tokenText = match[0];
    // Simulate BPE by splitting long words
    if (tokenText.length > 4 && !tokenText.match(/^\s+$/)) {
      // Split into ~4 char chunks (approximating BPE)
      for (let i = 0; i < tokenText.length; i += 4) {
        const chunk = tokenText.slice(i, i + 4);
        tokens.push({
          id: hashCode(chunk),
          text: chunk,
          position: position++,
          byteOffset: byteOffset,
        });
        byteOffset += new TextEncoder().encode(chunk).length;
      }
    } else {
      tokens.push({
        id: hashCode(tokenText),
        text: tokenText,
        position: position++,
        byteOffset: byteOffset,
      });
      byteOffset += new TextEncoder().encode(tokenText).length;
    }
  }

  return tokens;
}

/** Simple string hash code */
function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

/**
 * Semops operator patterns for AUOM detection
 */
const SEMOPS_PATTERNS: Record<string, RegExp> = {
  CKIN: /\b(ckin|dkin|chkin|checking\s+in)\b/i,
  ITERATE: /\b(iterate|oiterate)\b/i,
  REALAGENTS: /\b(realagents?|dispatch)\b/i,
  PBFLUSH: /\b(pbflush|flush)\b/i,
  PBDEEP: /\b(pbdeep|deep\s+analysis)\b/i,
  PBASSIMILATE: /\b(pbassimilate|pbassimilation)\b/i,
  PLURIBUS: /\b(pluribus)\b/i,
  BEAM: /\b(beam|log\s+entry)\b/i,
};

/**
 * Category detection patterns
 */
const CATEGORY_PATTERNS: Record<AUOMCategory, RegExp> = {
  temporal: /\b(now|today|tomorrow|yesterday|always|never|before|after|during|since|until|when|while|\d{4}-\d{2}-\d{2}|\d+\s*(ms|sec|min|hour|day)s?)\b/i,
  quantifier: /\b(\d+|one|two|three|four|five|ten|hundred|all|some|none|many|few|most|every)\b/i,
  spatial: /\b(here|there|where|above|below|inside|outside|near|far|between|through|across)\b/i,
  connector: /\b(and|or|but|if|then|because|therefore|however|although|unless|while)\b/i,
  entity: /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b/, // Proper nouns
  action: /\b(run|execute|create|delete|update|send|receive|process|analyze|validate)\b/i,
  relation: /\b(is|has|contains|belongs|connects|relates|depends)\b/i,
  modifier: /\b(very|quite|extremely|slightly|somewhat|completely)\b/i,
  operator: /(?:)/, // Handled separately via SEMOPS_PATTERNS
};

// =============================================================================
// MAIN CLIENT CLASS
// =============================================================================

export class TokenGeometryClient {
  private channelDim: number;
  private nsphereDim: number;

  constructor(options: { channelDim?: number; nsphereDim?: number } = {}) {
    this.channelDim = options.channelDim ?? 64;
    this.nsphereDim = options.nsphereDim ?? 128;
  }

  /**
   * Count tokens in text (browser approximation)
   */
  countTokens(text: string): number {
    return tokenizeSimple(text).length;
  }

  /**
   * Truncate text to max tokens
   */
  truncateToTokens(text: string, maxTokens: number): string {
    const tokens = tokenizeSimple(text);
    if (tokens.length <= maxTokens) return text;
    return tokens
      .slice(0, maxTokens)
      .map((t) => t.text)
      .join('');
  }

  /**
   * Analyze text for AUOM units
   */
  analyzeAUOM(text: string): AUOMUnit[] {
    const units: AUOMUnit[] = [];
    const seen = new Set<string>();

    // Detect semops operators
    for (const [opName, pattern] of Object.entries(SEMOPS_PATTERNS)) {
      let match: RegExpExecArray | null;
      const regex = new RegExp(pattern.source, 'gi');
      while ((match = regex.exec(text)) !== null) {
        const key = `${match.index}-${match[0]}`;
        if (!seen.has(key)) {
          seen.add(key);
          units.push({
            id: `auom-${hashCode(key).toString(16)}`,
            text: match[0],
            category: 'operator',
            confidence: 0.95,
            span: [match.index, match.index + match[0].length],
            metadata: { operator: opName },
          });
        }
      }
    }

    // Detect other categories
    for (const [category, pattern] of Object.entries(CATEGORY_PATTERNS)) {
      if (category === 'operator') continue;

      let match: RegExpExecArray | null;
      const regex = new RegExp(pattern.source, 'gi');
      while ((match = regex.exec(text)) !== null) {
        const key = `${match.index}-${match[0]}`;
        if (!seen.has(key)) {
          // Check not overlapping with existing
          const overlaps = units.some(
            (u) => match!.index >= u.span[0] && match!.index < u.span[1]
          );
          if (!overlaps) {
            seen.add(key);
            units.push({
              id: `auom-${hashCode(key).toString(16)}`,
              text: match[0],
              category: category as AUOMCategory,
              confidence: 0.8,
              span: [match.index, match.index + match[0].length],
            });
          }
        }
      }
    }

    // Sort by position
    return units.sort((a, b) => a.span[0] - b.span[0]);
  }

  /**
   * Encode text into sextet vector
   */
  async encodeSextet(text: string, auomUnits?: AUOMUnit[]): Promise<SextetVector> {
    const units = auomUnits ?? this.analyzeAUOM(text);

    // Semantic channel - base embedding
    const semantic = await hashEmbed(`semantic:${text}`, this.channelDim);

    // Syntactic channel - structure features
    const syntactic = this.encodeSyntactic(text);

    // Pragmatic channel - context/usage
    const pragmatic = this.encodePragmatic(text, units);

    // Temporal channel - time features
    const temporal = this.encodeTemporal(text, units);

    // Modal channel - modality indicators
    const modal = this.encodeModal(text);

    // Affective channel - sentiment
    const affective = this.encodeAffective(text);

    return { semantic, syntactic, pragmatic, temporal, modal, affective };
  }

  private encodeSyntactic(text: string): number[] {
    const vec = new Array(this.channelDim).fill(0);

    vec[0] = text.split(/\s+/).length; // Word count
    vec[1] = text.length; // Char count
    vec[2] = (text.match(/[.!?]/g) || []).length; // Sentence count
    vec[3] = text.includes('?') ? 1 : 0; // Question
    vec[4] = text.endsWith('!') || /^[A-Z]/.test(text) ? 1 : 0; // Imperative
    vec[5] = (text.match(/[()[\]]/g) || []).length; // Parenthetical
    vec[6] = (text.match(/[.,;:!?]/g) || []).length / Math.max(1, text.length); // Punct density

    return this.normalize(vec);
  }

  private encodePragmatic(text: string, units: AUOMUnit[]): number[] {
    const vec = new Array(this.channelDim).fill(0);

    // Count AUOM categories
    const categories: AUOMCategory[] = [
      'entity', 'action', 'relation', 'modifier', 'quantifier',
      'temporal', 'spatial', 'operator', 'connector',
    ];

    categories.forEach((cat, i) => {
      if (i < this.channelDim) {
        vec[i] = units.filter((u) => u.category === cat).length;
      }
    });

    return this.normalize(vec);
  }

  private encodeTemporal(text: string, units: AUOMUnit[]): number[] {
    const vec = new Array(this.channelDim).fill(0);

    const temporalWords = [
      'now', 'then', 'before', 'after', 'always', 'never',
      'today', 'tomorrow', 'yesterday', 'soon', 'later',
    ];

    const textLower = text.toLowerCase();
    temporalWords.forEach((word, i) => {
      if (i < this.channelDim && textLower.includes(word)) {
        vec[i] = 1;
      }
    });

    // Count temporal AUOM units
    const temporalCount = units.filter((u) => u.category === 'temporal').length;
    vec[Math.min(temporalWords.length, this.channelDim - 1)] = temporalCount;

    return this.normalize(vec);
  }

  private encodeModal(text: string): number[] {
    const vec = new Array(this.channelDim).fill(0);

    vec[0] = /```|def |class |function |import |const |let |var /.test(text) ? 1 : 0; // Code
    vec[1] = /https?:\/\//.test(text) ? 1 : 0; // URL
    vec[2] = /\/\w+\/\w+|\\w+\\w+/.test(text) ? 1 : 0; // Path
    vec[3] = /\{["\']?\w+["\']?\s*:/.test(text) ? 1 : 0; // JSON
    vec[4] = /[+\-*/=<>]{2,}|\d+\.\d+/.test(text) ? 1 : 0; // Math
    vec[5] = /^\s*[-*]\s/m.test(text) ? 1 : 0; // List

    return this.normalize(vec);
  }

  private encodeAffective(text: string): number[] {
    const vec = new Array(this.channelDim).fill(0);

    const positive = ['good', 'great', 'excellent', 'success', 'happy', 'wonderful', 'amazing', 'perfect'];
    const negative = ['bad', 'error', 'fail', 'wrong', 'problem', 'issue', 'broken', 'critical'];

    const textLower = text.toLowerCase();
    const posCount = positive.filter((w) => textLower.includes(w)).length;
    const negCount = negative.filter((w) => textLower.includes(w)).length;

    vec[0] = posCount;
    vec[1] = negCount;
    vec[2] = posCount - negCount; // Valence
    vec[3] = text.includes('!') ? 1 : 0; // Intensity
    vec[4] = (text.match(/[!?]/g) || []).length; // Arousal

    return this.normalize(vec);
  }

  private normalize(vec: number[]): number[] {
    const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
    return norm > 1e-10 ? vec.map((v) => v / norm) : vec;
  }

  /**
   * Project sextet vector to n-sphere
   */
  projectToNSphere(sextet: SextetVector): NSpherePoint {
    // Concatenate all channels
    const fullVector = [
      ...sextet.semantic,
      ...sextet.syntactic,
      ...sextet.pragmatic,
      ...sextet.temporal,
      ...sextet.modal,
      ...sextet.affective,
    ];

    // Random projection to n-sphere dimension
    const projected = this.randomProject(fullVector, this.nsphereDim);

    // Normalize to unit sphere
    const norm = Math.sqrt(projected.reduce((sum, v) => sum + v * v, 0));
    const coords = norm > 1e-10 ? projected.map((v) => v / norm) : projected;

    return { coords, radius: 1.0 };
  }

  private randomProject(vec: number[], targetDim: number): number[] {
    if (vec.length === targetDim) return vec;

    if (vec.length > targetDim) {
      // Truncate
      return vec.slice(0, targetDim);
    }

    // Pad with zeros
    return [...vec, ...new Array(targetDim - vec.length).fill(0)];
  }

  /**
   * Compute geodesic distance between two n-sphere points
   */
  geodesicDistance(p1: NSpherePoint, p2: NSpherePoint): number {
    const dot = p1.coords.reduce((sum, v, i) => sum + v * (p2.coords[i] || 0), 0);
    const clampedDot = Math.max(-1, Math.min(1, dot / (p1.radius * p2.radius)));
    return Math.acos(clampedDot) * p1.radius;
  }

  /**
   * Spherical linear interpolation (SLERP)
   */
  slerp(p1: NSpherePoint, p2: NSpherePoint, t: number): NSpherePoint {
    const dot = Math.max(
      -1,
      Math.min(1, p1.coords.reduce((sum, v, i) => sum + v * (p2.coords[i] || 0), 0))
    );
    const theta = Math.acos(dot);

    if (theta < 1e-10) return p1;

    const sinTheta = Math.sin(theta);
    const a = Math.sin((1 - t) * theta) / sinTheta;
    const b = Math.sin(t * theta) / sinTheta;

    const coords = p1.coords.map((v, i) => a * v + b * (p2.coords[i] || 0));
    return { coords, radius: 1.0 };
  }

  /**
   * Create superpositional state
   */
  createSuperposition(
    states: string[],
    weights?: number[]
  ): SuperpositionalState {
    const w = weights ?? states.map(() => 1);
    const norm = Math.sqrt(w.reduce((sum, v) => sum + v * v, 0));
    const amplitudes = w.map((v) => ({
      real: v / norm,
      imag: 0,
    }));

    return { basisStates: states, amplitudes };
  }

  /**
   * Get probability of state in superposition
   */
  probability(superposition: SuperpositionalState, state: string): number {
    const idx = superposition.basisStates.indexOf(state);
    if (idx === -1) return 0;

    const amp = superposition.amplitudes[idx];
    return amp.real * amp.real + amp.imag * amp.imag;
  }

  /**
   * Collapse superposition to single state
   */
  collapse(superposition: SuperpositionalState): string {
    const probs = superposition.amplitudes.map(
      (a) => a.real * a.real + a.imag * a.imag
    );
    const r = Math.random();
    let cumulative = 0;

    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (r < cumulative) {
        return superposition.basisStates[i];
      }
    }

    return superposition.basisStates[superposition.basisStates.length - 1];
  }

  /**
   * Validate LTL formula over state sequence
   */
  validateLTL(
    formula: LTLFormula,
    states: SuperpositionalState[]
  ): { valid: boolean; confidence: number } {
    if (!formula.operator) {
      // Atomic proposition
      const prop = String(formula.operands[0]);
      if (states.length === 0) return { valid: false, confidence: 0 };
      const prob = this.probability(states[states.length - 1], prop);
      return { valid: prob > 0.5, confidence: prob };
    }

    switch (formula.operator) {
      case 'always': {
        const prop = String(formula.operands[0]);
        const probs = states.map((s) => this.probability(s, prop));
        const minProb = Math.min(...probs);
        return { valid: minProb > 0.5, confidence: minProb };
      }

      case 'eventually': {
        const prop = String(formula.operands[0]);
        const probs = states.map((s) => this.probability(s, prop));
        const maxProb = Math.max(...probs);
        return { valid: maxProb > 0.5, confidence: maxProb };
      }

      case 'until': {
        const p = String(formula.operands[0]);
        const q = String(formula.operands[1]);
        let qFound = false;
        let pHeld = true;
        let confidence = 1.0;

        for (const state of states) {
          const qProb = this.probability(state, q);
          const pProb = this.probability(state, p);

          if (qProb > 0.5) {
            qFound = true;
            confidence = Math.min(confidence, qProb);
            break;
          }

          if (pProb <= 0.5) {
            pHeld = false;
            confidence = Math.min(confidence, pProb);
            break;
          }

          confidence = Math.min(confidence, pProb);
        }

        return { valid: qFound && pHeld, confidence };
      }

      case 'not': {
        const inner = formula.operands[0];
        const innerFormula =
          typeof inner === 'string'
            ? { operator: null, operands: [inner] }
            : inner;
        const result = this.validateLTL(innerFormula as LTLFormula, states);
        return { valid: !result.valid, confidence: 1 - result.confidence };
      }

      case 'and': {
        const r1 = this.validateLTL(
          typeof formula.operands[0] === 'string'
            ? { operator: null, operands: [formula.operands[0]] }
            : (formula.operands[0] as LTLFormula),
          states
        );
        const r2 = this.validateLTL(
          typeof formula.operands[1] === 'string'
            ? { operator: null, operands: [formula.operands[1]] }
            : (formula.operands[1] as LTLFormula),
          states
        );
        return {
          valid: r1.valid && r2.valid,
          confidence: Math.min(r1.confidence, r2.confidence),
        };
      }

      case 'or': {
        const r1 = this.validateLTL(
          typeof formula.operands[0] === 'string'
            ? { operator: null, operands: [formula.operands[0]] }
            : (formula.operands[0] as LTLFormula),
          states
        );
        const r2 = this.validateLTL(
          typeof formula.operands[1] === 'string'
            ? { operator: null, operands: [formula.operands[1]] }
            : (formula.operands[1] as LTLFormula),
          states
        );
        return {
          valid: r1.valid || r2.valid,
          confidence: Math.max(r1.confidence, r2.confidence),
        };
      }

      default:
        return { valid: false, confidence: 0 };
    }
  }

  /**
   * Full processing pipeline
   */
  async process(text: string): Promise<TokenGeometryResult> {
    const start = performance.now();

    // Tokenize
    const bpeTokens = tokenizeSimple(text);
    const tokenCount = bpeTokens.length;

    // AUOM analysis
    const auomUnits = this.analyzeAUOM(text);
    const detectedOperators = auomUnits
      .filter((u) => u.category === 'operator')
      .map((u) => u.metadata?.operator as string)
      .filter(Boolean);

    // Sextet encoding
    const sextetVector = await this.encodeSextet(text, auomUnits);

    // N-sphere projection
    const nspherePoint = this.projectToNSphere(sextetVector);

    const processingTimeMs = performance.now() - start;

    return {
      rawText: text,
      bpeTokens,
      tokenCount,
      auomUnits,
      detectedOperators,
      sextetVector,
      nspherePoint,
      ltlFormula: null,
      ltlValid: true,
      superposition: null,
      processingTimeMs,
      engineVersion: '1.0.0',
    };
  }

  /**
   * Convert bus events to superpositional states
   */
  busEventsToStates(events: Array<{ topic?: string; kind?: string; level?: string }>): SuperpositionalState[] {
    return events.map((event) => {
      const topic = event.topic || 'unknown';
      const kind = event.kind || 'event';
      const level = event.level || 'info';

      const basis = [
        `topic:${topic}`,
        `kind:${kind}`,
        `level:${level}`,
        'active',
      ];

      const weights = [1.0, 0.8, 0.6, level !== 'error' ? 1.0 : 0.3];

      return this.createSuperposition(basis, weights);
    });
  }

  /**
   * Emit processing result to bus
   */
  toBusEvent(result: TokenGeometryResult): Record<string, unknown> {
    return {
      id: crypto.randomUUID(),
      ts: Date.now() / 1000,
      iso: new Date().toISOString(),
      topic: 'token_geometry.process',
      kind: 'metric',
      level: 'info',
      actor: 'token_geometry_client',
      data: {
        token_count: result.tokenCount,
        auom_count: result.auomUnits.length,
        detected_operators: result.detectedOperators,
        ltl_valid: result.ltlValid,
        processing_time_ms: result.processingTimeMs,
        nsphere_coords: result.nspherePoint?.coords.slice(0, 8),
      },
    };
  }
}

// =============================================================================
// FACTORY & SINGLETON
// =============================================================================

let _clientInstance: TokenGeometryClient | null = null;

export function getTokenGeometryClient(
  options?: { channelDim?: number; nsphereDim?: number }
): TokenGeometryClient {
  if (!_clientInstance) {
    _clientInstance = new TokenGeometryClient(options);
  }
  return _clientInstance;
}

// =============================================================================
// HELPER FUNCTIONS FOR LTL FORMULA CONSTRUCTION
// =============================================================================

export const LTL = {
  always: (prop: string): LTLFormula => ({ operator: 'always', operands: [prop] }),
  eventually: (prop: string): LTLFormula => ({ operator: 'eventually', operands: [prop] }),
  next: (prop: string): LTLFormula => ({ operator: 'next', operands: [prop] }),
  until: (p: string, q: string): LTLFormula => ({ operator: 'until', operands: [p, q] }),
  and: (a: LTLFormula | string, b: LTLFormula | string): LTLFormula => ({ operator: 'and', operands: [a, b] }),
  or: (a: LTLFormula | string, b: LTLFormula | string): LTLFormula => ({ operator: 'or', operands: [a, b] }),
  not: (a: LTLFormula | string): LTLFormula => ({ operator: 'not', operands: [a] }),
  implies: (a: LTLFormula | string, b: LTLFormula | string): LTLFormula => ({ operator: 'implies', operands: [a, b] }),
  atom: (prop: string): LTLFormula => ({ operator: null, operands: [prop] }),
};

export default TokenGeometryClient;
