#!/bin/bash
# commit_ark_full.sh - Commit complete ARK implementation to git

set -e

cd /Users/kroma/pluribus

echo "Adding ARK package to git..."
git add nucleus/ark/
git add nucleus/tools/simulated_swarm_logs/ark/
git add MANIFEST.yaml

echo "Creating commit..."
git commit -m "feat: ARK Complete - Autonomous Reactive Kernel (26 files, 6 phases)

Implemented the complete ARK system - a negentropic evolutionary VCS that
unifies all Pluribus holons into a single evolution-aware version control.

## Package Structure (26 Python files)

### Core Module
- repository.py: ArkRepository with Cell Cycle (G1-S-G2-M)
- context.py: ArkCommitContext with DNA metadata
- inception.py: Self-bootstrapping and G1-G6 guard ladder
- integration.py: Bus, OHM, LASER, PBTSO connections

### Gates Module (DNA Triplet)
- inertia.py: Stability preservation (□ safety)
- entelecheia.py: Purpose enforcement (◇ liveness)
- homeostasis.py: System stability (homeostasis)

### Ribosome Module
- gene.py: Fundamental unit with etymology
- clade.py: Evolutionary branch with Thompson Sampling
- genome.py: OrganismGenome with Constitution

### Rhizom Module
- dag.py: Semantic commit DAG
- etymology.py: Semantic origin extraction
- lineage.py: Ancestry and CMP trajectory

### Portal Module
- ingest.py: Entropic source processing
- distill.py: Full negentropic transformation
- layers.py: 3-Layer architecture (Raw→Curated→Core)

### Synthesis Module
- ltl_spec.py: Linear Temporal Logic specifications
- grammar.py: SyGuS grammar-guided synthesis

## CLI Commands (8)
- ark init: Initialize repository
- ark status: Show DNA-aware status
- ark commit: Cell Cycle gated commit
- ark log: Show log with CMP
- ark health: System health check
- ark distill: Run distillation pipeline
- ark ancestry: Show lineage trajectory
- ark verify: LTL spec verification

## Key Features
- Cell Cycle Commits (G1→S→G2→M)
- Thompson Sampling for clade selection
- 8-dimensional H* entropy tracking
- LTL formalized DNA axioms
- Witness/attestation protocol
- Guard Ladder (G1-G6)
- PBTSO/OHM integration hooks

## Multi-Agent Swarm Synthesis
- Claude Opus: LTL/constitutional formalization
- Gemini 3 Pro: CMP/entropy optimization
- Codex 5.2: Implementation engineering

All 6 ARK phases complete (21-26, 150 steps)"

echo "Commit created successfully!"
echo "Run 'git push vps main' and 'git push origin main' to sync."
