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
 * - EditTripletExtractor: Extract (pre, diff, post) training data
 * - BranchReconciler: Git branch analysis and merge conflict prediction
 * - AleatoricCheck: Identify irreducible vs reducible uncertainty
 * - FourDCompiler: 4D code compilation (spatial + temporal)
 */

// Core temporal analysis
export * from './git-walker.js';
export * from './temporal-signal.js';
export * from './thrash-detector.js';
export * from './blame-map.js';
export * from './entelechy.js';
export * from './edit-triplet-extractor.js';

// Advanced temporal analysis (Steps 22-25)
export * from './branch-reconciler.js';
export * from './aleatoric-check.js';
export * from './four-d-compiler.js';
