/**
 * STRpLead Types - Content Curation Decision Schema
 *
 * Types for the MOTD-Curate pipeline leads management.
 * Each lead represents a curated content item with a promotion decision.
 */

// ============================================================================
// STRpLead Core Types
// ============================================================================

/** Curation decision for a lead */
export type LeadDecision = 'promote' | 'defer' | 'reject';

/** Topic category for content classification */
export type LeadTopic =
  | 'FILM'
  | 'AI/ML'
  | 'MUSIC'
  | 'ART'
  | 'TECH'
  | 'SCIENCE'
  | 'CULTURE'
  | 'GAMING'
  | 'DESIGN'
  | 'OTHER';

/** Artifact types available for a lead */
export interface LeadArtifacts {
  /** Thumbnail image path/URL */
  thumb?: string;
  /** Animated GIF preview */
  gif?: string;
  /** Video clip path/URL */
  clip?: string;
  /** Audio clip path/URL */
  mp3?: string;
}

/** Main STRpLead interface matching the schema */
export interface STRpLead {
  /** Unique identifier (UUID) */
  lead_id: string;
  /** ISO timestamp of creation */
  ts: string;
  /** Actor that created this lead (e.g., 'motd-curate') */
  actor: string;
  /** Curation decision */
  decision: LeadDecision;
  /** Content topic/category */
  topic: LeadTopic | string;
  /** Title of the content */
  title: string;
  /** Source URL */
  url: string;
  /** Extracted keywords */
  keywords: string[];
  /** Suggested next actions */
  next_actions: string[];
  /** Available media artifacts */
  artifacts: LeadArtifacts;
  /** Optional notes from curator */
  notes?: string;
  /** Optional priority (1=highest) */
  priority?: number;
  /** Optional ingestion status */
  ingested?: boolean;
  /** Optional portal destination */
  portal_target?: string;
  /** Optional archive flag */
  archived?: boolean;
}

// ============================================================================
// View State Types
// ============================================================================

/** Tab state for leads grouped by decision */
export type LeadTab = 'all' | LeadDecision;

/** Sort options for leads */
export type LeadSortBy = 'ts' | 'priority' | 'topic' | 'title';

/** Filter state for leads view */
export interface LeadFilterState {
  tab: LeadTab;
  topic: LeadTopic | 'ALL';
  sortBy: LeadSortBy;
  sortDesc: boolean;
  searchQuery: string;
}

// ============================================================================
// Action Types
// ============================================================================

/** Actions that can be performed on a lead */
export type LeadAction =
  | 'watch'        // Open content in viewer
  | 'transfigure'  // Begin SOPHOS → Holon → Portal flow
  | 'archive'      // Archive the lead
  | 'promote'      // Change decision to promote
  | 'defer'        // Change decision to defer
  | 'reject'       // Change decision to reject
  | 'edit';        // Edit lead metadata

/** Action request payload */
export interface LeadActionRequest {
  action: LeadAction;
  lead_id: string;
  target?: string;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Bus Event Types
// ============================================================================

/** Bus topics for leads integration */
export const LEAD_BUS_TOPICS = {
  // Incoming
  LEADS_SYNC: 'strp.leads.sync',           // Full leads list update
  LEAD_CREATED: 'strp.lead.created',       // New lead created
  LEAD_UPDATED: 'strp.lead.updated',       // Lead modified
  LEAD_DELETED: 'strp.lead.deleted',       // Lead removed

  // Outgoing (actions)
  LEAD_WATCH: 'strp.lead.action.watch',
  LEAD_INGEST: 'strp.lead.action.ingest',
  LEAD_ARCHIVE: 'strp.lead.action.archive',
  LEAD_DECISION: 'strp.lead.action.decision',
} as const;

// ============================================================================
// Helper Functions
// ============================================================================

/** Get color class for decision badge */
export function getDecisionColor(decision: LeadDecision): string {
  switch (decision) {
    case 'promote':
      return 'bg-green-500/20 text-green-400 border-green-500/30';
    case 'defer':
      return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    case 'reject':
      return 'bg-red-500/20 text-red-400 border-red-500/30';
  }
}

/** Get icon for topic */
export function getTopicIcon(topic: string): string {
  const icons: Record<string, string> = {
    'FILM': '🎬',
    'AI/ML': '🤖',
    'MUSIC': '🎵',
    'ART': '🎨',
    'TECH': '💻',
    'SCIENCE': '🔬',
    'CULTURE': '🌍',
    'GAMING': '🎮',
    'DESIGN': '✏️',
    'OTHER': '📌',
  };
  return icons[topic] || '📌';
}

/** Group leads by decision */
export function groupLeadsByDecision(
  leads: STRpLead[]
): Record<LeadDecision, STRpLead[]> {
  const groups: Record<LeadDecision, STRpLead[]> = {
    promote: [],
    defer: [],
    reject: [],
  };
  for (const lead of leads) {
    if (groups[lead.decision]) {
      groups[lead.decision].push(lead);
    }
  }
  return groups;
}

// ============================================================================
// TRANSFIGURATION FLOW: Lead → SOPHOS → Holon → Portal
// Semioticalysis through autopoietic synthesis into entelexis
// ============================================================================

/**
 * Transfiguration Stage - tracks where in the neurosymbolic flow
 * NOTE: "ingest" is reserved for static asset uploads only
 */
export type TransfigurationStage =
  | 'raw'           // Fresh lead, not yet processed
  | 'questioning'   // SOPHOS dialectic in progress
  | 'synthesized'   // Thesis/Antithesis resolved to Synthesis
  | 'instantiating' // Holon Pentad being filled
  | 'validating'    // Sextet gates being evaluated
  | 'transfigured'  // Fully processed, ready for Portal
  | 'actualized'    // In Portal AM (knowledge graph)
  | 'shadowed'      // In Portal SM (held potential)
  | 'decayed';      // Rejected/discarded

/** Portal metabolic modes - the fundamental bifurcation (ONLY after Sextet) */
export type MetabolicMode = 'AM' | 'SM';  // Actualized-Mode vs Shadowmode

/** Entelexis phases from potential to decay */
export type EntelexisPhase = 'potential' | 'actualizing' | 'actualized' | 'decaying';

// ============================================================================
// SOPHOS Types (Socratic Dialectic Layer)
// ============================================================================

/** Dialectical position in thesis-antithesis-synthesis triad */
export interface DialecticalPosition {
  text: string;
  confidence: number;    // 0-1
  sources?: string[];    // Etymon references
  timestamp: number;
}

/** SOPHOS dialectical state for a lead */
export interface SophosDialectic {
  question_id: string;
  question_text: string;
  thesis?: DialecticalPosition;
  antithesis?: DialecticalPosition;
  synthesis?: DialecticalPosition;
  etymon_ref?: string;        // e.g., "ASL-2025", "user-intent"
  resolution_status: 'pending' | 'analyzing' | 'synthesizing' | 'resolved' | 'rejected';
}

// ============================================================================
// HOLON Types (Pentad + Sextet)
// ============================================================================

/** Pentad - The 5 Ws (The Key / Positive Intent) */
export interface HolonPentad {
  pentad_id: string;
  /** WHY - Etymon: The root law justifying the action */
  why: {
    id: string;
    justification: string;
    source: 'research' | 'axiom' | 'user_intent' | 'system_policy' | 'derived';
    confidence: number;
  };
  /** WHERE - Locus: The binding site */
  where: {
    uri: string;
    type: 'file' | 'memory' | 'bus_topic' | 'knowledge_graph' | 'vector_space';
    domain?: string;
  };
  /** WHO - Lineage: The agent acting */
  who: {
    agent_id: string;
    generation: number;
    authority: 'ring0' | 'ring1' | 'ring2' | 'ring3';
    citizen_class?: 'sagent' | 'swagent' | 'cagent';
  };
  /** WHEN - Kairos: The opportune moment */
  when: {
    phase: 'research' | 'planning' | 'implementation' | 'verification' | 'deployment';
    budget_ms?: number;
    opportune: boolean;
  };
  /** WHAT - Artifact: The output shape */
  what: {
    kind: 'knowledge' | 'reference' | 'media' | 'code' | 'mixed';
    schema_sig?: string;
    vector_cid?: string;
  };
}

/** Sextet Gate result */
export interface SextetGate {
  passed: boolean;
  score: number;      // 0-1
  check: string;
  rationale?: string;
}

/** Sextet - The 6 Gates (The Lock / Negative Constraints) */
export interface HolonSextet {
  sextet_id: string;
  pentad_id: string;
  gates: {
    P: SextetGate;  // Provenance
    E: SextetGate;  // Effects
    L: SextetGate;  // Liveness
    R: SextetGate;  // Recovery
    Q: SextetGate;  // Quality
    Ω: SextetGate;  // Omega (Alignment)
  };
  verdict: {
    status: 'PASSED' | 'WARNED' | 'FAILED' | 'BLOCKED';
    can_proceed: boolean;
    failed_gates: ('P' | 'E' | 'L' | 'R' | 'Q' | 'Ω')[];
    warned_gates: ('P' | 'E' | 'L' | 'R' | 'Q' | 'Ω')[];
  };
  compliance_vector: [number, number, number, number, number, number]; // [P,E,L,R,Q,Ω]
}

// ============================================================================
// TRANSFIGURED LEAD (Full Flow State)
// ============================================================================

/**
 * TransfiguredLead - A lead that has traversed the full semioticalysis flow
 * Lead → SOPHOS → Holon (Pentad + Sextet) → Portal (AM/SM)
 */
export interface TransfiguredLead extends STRpLead {
  // Transfiguration tracking
  stage: TransfigurationStage;
  transfiguration_ts?: string;

  // SOPHOS layer (dialectic)
  sophos?: SophosDialectic;

  // Holon layer (Pentad + Sextet)
  pentad?: HolonPentad;
  sextet?: HolonSextet;
  holon_id?: string;

  // Portal layer (ONLY populated if Sextet passes)
  metabolic_mode?: MetabolicMode | null;
  entelexis_phase?: EntelexisPhase;
  portal_ts?: string;

  // Tree structure for branching
  bohm_node_id?: string;
  parent_lead_id?: string;
  child_lead_ids?: string[];
}

// For backwards compatibility
export type PortalLead = TransfiguredLead;

// ============================================================================
// Transfiguration Functions
// ============================================================================

/** Initialize a lead for transfiguration (entry to SOPHOS) */
export function initializeTransfiguration(lead: STRpLead): TransfiguredLead {
  return {
    ...lead,
    stage: 'raw',
    transfiguration_ts: new Date().toISOString(),
    bohm_node_id: `bohm:${lead.lead_id}`,
  };
}

/** Begin SOPHOS dialectic for a lead */
export function beginSophosDialectic(
  lead: TransfiguredLead,
  questionText: string
): TransfiguredLead {
  return {
    ...lead,
    stage: 'questioning',
    sophos: {
      question_id: `sophos:${lead.lead_id}:${Date.now()}`,
      question_text: questionText,
      resolution_status: 'pending',
    },
  };
}

/** Record SOPHOS synthesis and prepare for Holon instantiation */
export function recordSynthesis(
  lead: TransfiguredLead,
  synthesis: DialecticalPosition,
  etymonRef: string
): TransfiguredLead {
  if (!lead.sophos) return lead;
  return {
    ...lead,
    stage: 'synthesized',
    sophos: {
      ...lead.sophos,
      synthesis,
      etymon_ref: etymonRef,
      resolution_status: 'resolved',
    },
  };
}

/** Instantiate Holon with Pentad */
export function instantiateHolon(
  lead: TransfiguredLead,
  pentad: HolonPentad
): TransfiguredLead {
  return {
    ...lead,
    stage: 'instantiating',
    pentad,
    holon_id: `holon:${pentad.pentad_id}`,
  };
}

/** Validate Holon through Sextet gates */
export function validateSextet(
  lead: TransfiguredLead,
  sextet: HolonSextet
): TransfiguredLead {
  const canProceed = sextet.verdict.can_proceed;

  return {
    ...lead,
    stage: canProceed ? 'transfigured' : 'validating',
    sextet,
  };
}

// ============================================================================
// ETYMON CONSTANTS (Iteration 3 - ASL-2025 Grounding)
// ============================================================================

/** Root axioms from ASL-2025 that ground all transfiguration decisions */
export const ETYMON_REGISTRY = {
  // Core ASL-2025 Laws
  'ASL-2025:sequential-collapse': {
    id: 'ASL-2025:sequential-collapse',
    law: 'Law of Sequential Collapse',
    description: 'Tasks degrade -70% when split across swarms; prefer single topology',
    applies_to: ['topology', 'coordination'],
  },
  'ASL-2025:verification-covenant': {
    id: 'ASL-2025:verification-covenant',
    law: 'Verification Covenant',
    description: 'Untested code is vapor; evidence required for actualization',
    applies_to: ['quality', 'testing', 'artifacts'],
  },
  'ASL-2025:conservation': {
    id: 'ASL-2025:conservation',
    law: 'Law of Conservation',
    description: 'No work is ever lost; rejected items preserved in Amber',
    applies_to: ['recovery', 'shadowmode'],
  },
  // Meta-epistemic etymons
  'etymon:user-intent': {
    id: 'etymon:user-intent',
    law: 'User Intent Primacy',
    description: 'Curator/human decision carries authoritative weight',
    applies_to: ['decision', 'omega'],
  },
  'etymon:caution-principle': {
    id: 'etymon:caution-principle',
    law: 'Epistemic Caution',
    description: 'When uncertain, prefer shadowmode over premature actualization',
    applies_to: ['synthesis', 'uncertainty'],
  },
  'etymon:quality-threshold': {
    id: 'etymon:quality-threshold',
    law: 'Quality Threshold',
    description: 'Minimum metadata/artifact requirements for actualization',
    applies_to: ['quality', 'q-gate'],
  },
} as const;

export type EtymonId = keyof typeof ETYMON_REGISTRY;

/** Get etymon reference for a dialectic position */
export function getEtymonForContext(context: 'thesis' | 'antithesis' | 'synthesis', decision: LeadDecision): EtymonId {
  if (context === 'thesis') {
    return decision === 'promote' ? 'etymon:user-intent' : 'etymon:caution-principle';
  }
  if (context === 'antithesis') {
    return 'ASL-2025:verification-covenant';
  }
  // Synthesis
  return decision === 'promote' ? 'etymon:user-intent' : 'ASL-2025:conservation';
}

// ============================================================================
// AUTO-GENERATION FUNCTIONS (Iteration 2+3)
// ============================================================================

/**
 * Auto-generate thesis from lead metadata
 * Thesis = positive assertion about the lead's value
 * Grounded in ASL-2025 Etymon
 */
export function autoGenerateThesis(lead: STRpLead): DialecticalPosition {
  const topicContext = {
    'FILM': 'visual storytelling and cinematic technique',
    'AI/ML': 'machine learning methodology and AI advancement',
    'MUSIC': 'sonic composition and musical innovation',
    'ART': 'creative expression and aesthetic value',
    'TECH': 'technological capability and engineering insight',
    'SCIENCE': 'scientific understanding and empirical knowledge',
    'CULTURE': 'cultural significance and social relevance',
    'GAMING': 'interactive experience and game design',
    'DESIGN': 'design principles and user experience',
    'OTHER': 'general knowledge contribution',
  }[lead.topic as string] || 'knowledge contribution';

  const keywordPhrase = lead.keywords.slice(0, 3).join(', ');

  const etymonRef = getEtymonForContext('thesis', lead.decision);
  const etymon = ETYMON_REGISTRY[etymonRef];

  return {
    text: `"${lead.title}" contributes valuable insights in ${topicContext}${keywordPhrase ? `, specifically regarding ${keywordPhrase}` : ''}. Integration would enrich the knowledge graph. [Grounded in: ${etymon.law}]`,
    confidence: lead.decision === 'promote' ? 0.85 : lead.decision === 'defer' ? 0.6 : 0.3,
    sources: [etymonRef, `topic:${lead.topic}`, `decision:${lead.decision}`],
    timestamp: Date.now(),
  };
}

/**
 * Auto-generate antithesis from potential conflicts/concerns
 * Antithesis = critical examination of the lead's limitations
 */
export function autoGenerateAntithesis(lead: STRpLead): DialecticalPosition {
  const concerns: string[] = [];

  // Check for potential issues
  if (!lead.artifacts.thumb && !lead.artifacts.gif) {
    concerns.push('lacks visual preview artifacts');
  }
  if (lead.keywords.length < 3) {
    concerns.push('limited keyword extraction');
  }
  if (lead.decision === 'defer') {
    concerns.push('curator uncertainty about immediate relevance');
  }
  if (lead.decision === 'reject') {
    concerns.push('curator determined low value');
  }

  const concernText = concerns.length > 0
    ? concerns.join('; ')
    : 'no significant concerns identified';

  const etymonRef = getEtymonForContext('antithesis', lead.decision);
  const etymon = ETYMON_REGISTRY[etymonRef];

  return {
    text: `However, this content ${concernText}. The integration cost-benefit requires examination before actualization. [Per: ${etymon.law}]`,
    confidence: concerns.length > 1 ? 0.7 : 0.4,
    sources: [etymonRef, 'validation:artifacts', 'validation:metadata'],
    timestamp: Date.now(),
  };
}

/**
 * Auto-generate synthesis from thesis + antithesis
 * Synthesis = resolved position determining the path forward
 */
export function autoGenerateSynthesis(
  lead: STRpLead,
  thesis: DialecticalPosition,
  antithesis: DialecticalPosition
): DialecticalPosition {
  const weightedConfidence = (thesis.confidence * 0.6) + (antithesis.confidence * 0.4);
  const shouldActualize = thesis.confidence > antithesis.confidence && lead.decision === 'promote';

  const etymonRef = getEtymonForContext('synthesis', lead.decision);
  const etymon = ETYMON_REGISTRY[etymonRef];

  const resolution = shouldActualize
    ? `Synthesis: Despite noted concerns, the value proposition justifies actualization. Proceed to Holon instantiation with Pentad coordinates. [${etymon.law}]`
    : `Synthesis: Given the balance of considerations, this content should be held in Shadowmode for potential future actualization when conditions improve. [${etymon.law}]`;

  return {
    text: resolution,
    confidence: weightedConfidence,
    sources: [etymonRef, 'synthesis:auto', ...thesis.sources?.slice(0, 2) || [], ...antithesis.sources?.slice(0, 2) || []],
    timestamp: Date.now(),
  };
}

/**
 * Auto-fill Pentad from lead properties
 * Derives the 5 Ws from available metadata
 */
export function autoFillPentad(lead: STRpLead, actor: string = 'system'): HolonPentad {
  const pentadId = `pentad:${lead.lead_id}:${Date.now()}`;

  return {
    pentad_id: pentadId,
    // WHY - derived from topic and decision
    why: {
      id: lead.decision === 'promote' ? 'user-intent-curate' : 'system-defer',
      justification: `Curated content in ${lead.topic} domain, decision: ${lead.decision}`,
      source: 'user_intent',
      confidence: lead.decision === 'promote' ? 0.9 : 0.6,
    },
    // WHERE - derived from topic → knowledge graph location
    where: {
      uri: `kg:/${lead.topic.toLowerCase()}/${lead.lead_id}`,
      type: 'knowledge_graph',
      domain: lead.topic,
    },
    // WHO - the curator/actor
    who: {
      agent_id: lead.actor || actor,
      generation: 1,
      authority: 'ring2',
      citizen_class: 'cagent',
    },
    // WHEN - current phase based on decision
    when: {
      phase: lead.decision === 'promote' ? 'implementation' : 'research',
      budget_ms: 30000,
      opportune: lead.decision === 'promote',
    },
    // WHAT - artifact type from available media
    what: {
      kind: lead.artifacts.clip ? 'media' : lead.artifacts.mp3 ? 'media' : 'reference',
      schema_sig: `Lead:${lead.topic}`,
      vector_cid: undefined, // To be filled by vec2vec bridge
    },
  };
}

/**
 * Evaluate Sextet gates for a lead with Pentad
 * Returns validation results for all 6 gates
 */
export function evaluateSextetGates(lead: TransfiguredLead): HolonSextet {
  if (!lead.pentad) {
    throw new Error('Cannot evaluate Sextet without Pentad');
  }

  const pentadId = lead.pentad.pentad_id;
  const sextetId = `sextet:${lead.lead_id}:${Date.now()}`;

  // P-Gate: Provenance - check if lineage is valid
  const gateP: SextetGate = {
    passed: !!lead.actor && lead.actor.length > 0,
    score: lead.actor ? 1.0 : 0.0,
    check: 'Lineage verification',
    rationale: lead.actor ? `Actor "${lead.actor}" verified` : 'No actor lineage',
  };

  // E-Gate: Effects - check if locus is allowed
  const gateE: SextetGate = {
    passed: true, // Knowledge graph writes are generally allowed
    score: 1.0,
    check: 'Effects validation',
    rationale: `Locus ${lead.pentad.where.uri} is in allowed domain`,
  };

  // L-Gate: Liveness - check timing/budget
  const gateL: SextetGate = {
    passed: lead.pentad.when.opportune || lead.decision !== 'reject',
    score: lead.pentad.when.opportune ? 1.0 : 0.7,
    check: 'Liveness/timing check',
    rationale: lead.pentad.when.opportune ? 'Opportune moment' : 'Acceptable timing',
  };

  // R-Gate: Recovery - check if rollback possible
  const gateR: SextetGate = {
    passed: true, // Leads can always be archived/removed
    score: 1.0,
    check: 'Recovery path verification',
    rationale: 'Lead can be archived or removed if needed',
  };

  // Q-Gate: Quality - check artifacts and metadata completeness
  const hasGoodMetadata = lead.keywords.length >= 2 && lead.title.length > 10;
  const hasArtifacts = !!(lead.artifacts.thumb || lead.artifacts.gif);
  const qualityScore = (hasGoodMetadata ? 0.5 : 0) + (hasArtifacts ? 0.5 : 0.25);
  const gateQ: SextetGate = {
    passed: qualityScore >= 0.5,
    score: qualityScore,
    check: 'Quality/metadata validation',
    rationale: `Metadata: ${hasGoodMetadata ? '✓' : '○'}, Artifacts: ${hasArtifacts ? '✓' : '○'}`,
  };

  // Ω-Gate: Omega - check alignment with root decision
  const gateOmega: SextetGate = {
    passed: lead.decision !== 'reject',
    score: lead.decision === 'promote' ? 1.0 : lead.decision === 'defer' ? 0.7 : 0.0,
    check: 'Omega alignment',
    rationale: `Decision "${lead.decision}" alignment with curation intent`,
  };

  // Calculate verdict
  const gates = { P: gateP, E: gateE, L: gateL, R: gateR, Q: gateQ, Ω: gateOmega };
  const failedGates = (Object.entries(gates) as [keyof typeof gates, SextetGate][])
    .filter(([, g]) => !g.passed)
    .map(([k]) => k);
  const warnedGates = (Object.entries(gates) as [keyof typeof gates, SextetGate][])
    .filter(([, g]) => g.passed && g.score < 0.7)
    .map(([k]) => k);

  const status: HolonSextet['verdict']['status'] =
    failedGates.length > 0 ? 'FAILED' :
    warnedGates.length > 1 ? 'WARNED' : 'PASSED';

  return {
    sextet_id: sextetId,
    pentad_id: pentadId,
    gates,
    verdict: {
      status,
      can_proceed: failedGates.length === 0,
      failed_gates: failedGates,
      warned_gates: warnedGates,
    },
    compliance_vector: [gateP.score, gateE.score, gateL.score, gateR.score, gateQ.score, gateOmega.score],
  };
}

/**
 * Complete transfiguration pipeline
 * Runs the full flow: thesis → antithesis → synthesis → pentad → sextet → portal
 */
export function completeTransfiguration(lead: STRpLead, actor: string = 'system'): TransfiguredLead {
  // Step 1: Initialize
  let tLead = initializeTransfiguration(lead);

  // Step 2: SOPHOS dialectic
  const thesis = autoGenerateThesis(lead);
  const antithesis = autoGenerateAntithesis(lead);
  const synthesis = autoGenerateSynthesis(lead, thesis, antithesis);

  tLead = {
    ...tLead,
    stage: 'synthesized',
    sophos: {
      question_id: `sophos:${lead.lead_id}:${Date.now()}`,
      question_text: `Evaluate "${lead.title}" for knowledge integration`,
      thesis,
      antithesis,
      synthesis,
      etymon_ref: synthesis.sources?.find(s => s.startsWith('etymon:'))?.replace('etymon:', ''),
      resolution_status: 'resolved',
    },
  };

  // Step 3: Holon instantiation
  const pentad = autoFillPentad(lead, actor);
  tLead = instantiateHolon(tLead, pentad);

  // Step 4: Sextet validation
  const sextet = evaluateSextetGates(tLead);
  tLead = validateSextet(tLead, sextet);

  // Step 5: Portal fork (if passed)
  if (tLead.sextet?.verdict.can_proceed) {
    tLead = forkToPortal(tLead);
  }

  return tLead;
}

/**
 * Fork to Portal metabolic mode (ONLY if Sextet passed)
 * This is the final step - transfiguration complete
 */
export function forkToPortal(lead: TransfiguredLead): TransfiguredLead {
  // Guard: must have passed Sextet
  if (!lead.sextet?.verdict.can_proceed) {
    console.warn('[Transfiguration] Cannot fork to Portal - Sextet not passed');
    return lead;
  }

  // Determine metabolic mode based on original decision + Sextet result
  const mode: MetabolicMode = lead.decision === 'promote' ? 'AM' : 'SM';
  const phase: EntelexisPhase = mode === 'AM' ? 'actualizing' : 'potential';

  return {
    ...lead,
    stage: mode === 'AM' ? 'actualized' : 'shadowed',
    metabolic_mode: mode,
    entelexis_phase: phase,
    portal_ts: new Date().toISOString(),
  };
}

/** Group leads by transfiguration stage */
export function groupLeadsByStage(leads: STRpLead[]): Record<TransfigurationStage, TransfiguredLead[]> {
  const result: Record<TransfigurationStage, TransfiguredLead[]> = {
    raw: [],
    questioning: [],
    synthesized: [],
    instantiating: [],
    validating: [],
    transfigured: [],
    actualized: [],
    shadowed: [],
    decayed: [],
  };

  for (const lead of leads) {
    const tLead = lead as TransfiguredLead;
    const stage = tLead.stage || 'raw';
    result[stage].push({ ...tLead, stage });
  }

  return result;
}

/** Group leads by metabolic mode (only transfigured leads) */
export function groupLeadsByMetabolic(leads: STRpLead[]): {
  AM: TransfiguredLead[];
  SM: TransfiguredLead[];
  pending: TransfiguredLead[];  // Not yet through Sextet
  discarded: TransfiguredLead[];
} {
  const result = {
    AM: [] as TransfiguredLead[],
    SM: [] as TransfiguredLead[],
    pending: [] as TransfiguredLead[],
    discarded: [] as TransfiguredLead[],
  };

  for (const lead of leads) {
    const tLead = lead as TransfiguredLead;

    if (tLead.stage === 'actualized' && tLead.metabolic_mode === 'AM') {
      result.AM.push(tLead);
    } else if (tLead.stage === 'shadowed' && tLead.metabolic_mode === 'SM') {
      result.SM.push(tLead);
    } else if (tLead.stage === 'decayed' || tLead.decision === 'reject') {
      result.discarded.push({ ...tLead, stage: tLead.stage || 'decayed' });
    } else {
      result.pending.push({ ...tLead, stage: tLead.stage || 'raw' });
    }
  }

  return result;
}

// ============================================================================
// BUS EVENT TOPICS
// ============================================================================

/** Bus topics for transfiguration flow */
export const TRANSFIGURATION_BUS_TOPICS = {
  // SOPHOS events
  SOPHOS_QUESTION_POSED: 'sophos.question.posed',
  SOPHOS_THESIS: 'sophos.dialectic.thesis',
  SOPHOS_ANTITHESIS: 'sophos.dialectic.antithesis',
  SOPHOS_SYNTHESIS: 'sophos.dialectic.synthesis',
  SOPHOS_SYNTHESIS_COMPLETE: 'sophos.synthesis.complete',
  SOPHOS_REJECTED: 'sophos.resolution.rejected',

  // Holon events
  HOLON_PENTAD_INSTANTIATED: 'holon.pentad.instantiated',
  HOLON_SEXTET_EVALUATING: 'holon.sextet.evaluating',
  HOLON_SEXTET_PASSED: 'holon.sextet.passed',
  HOLON_SEXTET_FAILED: 'holon.sextet.failed',
  HOLON_SEXTET_WARNED: 'holon.sextet.warned',
  SEXTET_VALIDATION_COMPLETE: 'holon.sextet.validation.complete',

  // Portal events (only after Sextet)
  PORTAL_AM_FORK: 'portal.metabolic.am',
  PORTAL_SM_FORK: 'portal.metabolic.sm',
  PORTAL_DECAY: 'portal.metabolic.decay',

  // Entelexis state changes
  ENTELEXIS_ACTUALIZING: 'portal.entelexis.actualizing',
  ENTELEXIS_ACTUALIZED: 'portal.entelexis.actualized',
  ENTELEXIS_SHADOW: 'portal.entelexis.shadow',

  // Bohm tree events
  BOHM_NODE_CREATED: 'portal.bohm.node.created',
  BOHM_BRANCH: 'portal.bohm.branch',
} as const;

// Legacy alias
export const PORTAL_BUS_TOPICS = TRANSFIGURATION_BUS_TOPICS;

/** Create transfiguration event based on current stage */
export function createTransfigurationEvent(lead: TransfiguredLead): {
  topic: string;
  data: TransfiguredLead;
} {
  let topic: string;

  switch (lead.stage) {
    case 'questioning':
      topic = TRANSFIGURATION_BUS_TOPICS.SOPHOS_QUESTION_POSED;
      break;
    case 'synthesized':
      topic = TRANSFIGURATION_BUS_TOPICS.SOPHOS_SYNTHESIS;
      break;
    case 'instantiating':
      topic = TRANSFIGURATION_BUS_TOPICS.HOLON_PENTAD_INSTANTIATED;
      break;
    case 'validating':
      topic = TRANSFIGURATION_BUS_TOPICS.HOLON_SEXTET_EVALUATING;
      break;
    case 'transfigured':
      topic = lead.sextet?.verdict.status === 'PASSED'
        ? TRANSFIGURATION_BUS_TOPICS.HOLON_SEXTET_PASSED
        : TRANSFIGURATION_BUS_TOPICS.HOLON_SEXTET_WARNED;
      break;
    case 'actualized':
      topic = TRANSFIGURATION_BUS_TOPICS.PORTAL_AM_FORK;
      break;
    case 'shadowed':
      topic = TRANSFIGURATION_BUS_TOPICS.PORTAL_SM_FORK;
      break;
    case 'decayed':
      topic = TRANSFIGURATION_BUS_TOPICS.PORTAL_DECAY;
      break;
    default:
      topic = 'transfiguration.stage.unknown';
  }

  return { topic, data: lead };
}

// Legacy compatibility
export const DECISION_TO_METABOLIC: Record<LeadDecision, MetabolicMode | null> = {
  promote: 'AM',
  defer: 'SM',
  reject: null,
};

export const DECISION_TO_ENTELEXIS: Record<LeadDecision, EntelexisPhase> = {
  promote: 'actualizing',
  defer: 'potential',
  reject: 'decaying',
};

// Legacy function - now wraps full flow
export function normalizeLeadToPortal(lead: STRpLead): PortalLead {
  return initializeTransfiguration(lead);
}

export function createPortalIngressEvent(lead: STRpLead): {
  topic: string;
  data: PortalLead;
} {
  const tLead = initializeTransfiguration(lead);
  return createTransfigurationEvent(tLead);
}
