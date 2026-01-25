#!/bin/bash
# commit_distillation.sh - Commit Neural Distillation System to git

set -e

cd /Users/kroma/pluribus

echo "Adding distillation system files to git..."
git add nucleus/tools/distillation/
git add nucleus/tools/reactive_mutator/

echo "Creating commit..."
git commit -m "feat: Production Neural Distillation System

Implemented production-ready Negentropic Distillation System that transforms
entropic source repositories into negentropic target repositories using LTL-guided
reactive synthesis and deep learning-based thrash prediction.

Core Features:
- Neural Adapter with AST-based complexity analysis (McCabe cyclomatic complexity)
- Multi-dimensional feature extraction (complexity, depth, anti-patterns, churn)
- Triplet DNA Gates (Inertia, Entelecheia, Homeostasis) with isolation checks
- Distillation Engine CLI with file walking and copying to target
- Kill Switch for emergency abort
- Integration hooks for PluribusSystem events

Enhancements:
- AST depth calculation for detecting excessive nesting
- Anti-pattern detection (Manager, AbstractFactory, GodClass, etc.)
- Weighted thrash prediction model with multiple decision criteria
- Comprehensive error handling and structured logging
- Batch prediction support

Testing & Documentation:
- Unit tests for neural adapter, DNA gates, and end-to-end workflow
- User guide with quick start, use cases, and examples
- Validation script for demonstrating good/bad/complex code classification
- Three example files showcasing acceptance/rejection criteria

Architectural Improvements:
- Enhanced feature vector with 5 dimensions (vs previous simple metrics)
- Actual file synthesis to target directory (vs mock acceptance)
- Relative path preservation during distillation
- UTF-8 encoding support throughout

Related Artifacts:
- production_distillation_architecture.md
- neural_gate_design.md
- neural_evolution_roadmap.md
- enhancement_plan.md

Resolves aleatoric gaps in the distillation pipeline and prepares system for
production deployment and neural model training."

echo "Commit created successfully!"
echo "Run 'git push vps main' and 'git push origin main' to sync remotes."
