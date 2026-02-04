/**
 * @nesy/temporal - Git Archeology and Temporal Analysis
 *
 * Phase 2 of NeSy Evolution: Temporal Archeology
 *
 * Components:
 * - GitWalker: Fast git history traversal
 * - TemporalSignal: Commit frequency, author entropy, churn metrics
 * - ThrashDetector: Identify high-churn low-value code
 * - BlameMap: Map code to author archetypes
 * - EntelechyExtractor: Signal vs Noise identification
 */

export * from './git-walker';
export * from './temporal-signal';
export * from './thrash-detector';
export * from './blame-map';
export * from './entelechy';
