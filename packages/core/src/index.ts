/**
 * @nesy/core - Core primitives for neurosymbolic compilation
 *
 * Three layers:
 * - Neural: embedding, attention, adaptation
 * - Symbolic: terms, unification, rewriting
 * - Bridge: grounding, lifting, discretization
 */

export * from './types';
export * from './neural';
export * from './symbolic';
export * from './bridge';
export * from './context';

// LSA Pipeline (Phase 1: Epistemic Foundation)
export * from './tokenizer';
export * from './tfidf';
export * from './svd';
export * from './lsa';
export * from './semantic-collapser';
