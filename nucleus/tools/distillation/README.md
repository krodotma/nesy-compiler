# Distillation System User Guide

## Overview
The **Negentropic Distillation System** transforms "entropic" source repositories (containing code with high structural entropy, thrash, or bloat) into "negentropic" target repositories (clean, purposeful, stable code).

## Quick Start

### Basic Usage
```bash
cd /Users/kroma/pluribus
python3 -m nucleus.tools.distillation.distill_engine \
    --source /path/to/entropic/repo \
    --target /path/to/negentropic/repo
```

### With Neural Model
```bash
python3 -m nucleus.tools.distillation.distill_engine \
    --source /path/to/entropic/repo \
    --target /path/to/negentropic/repo \
    --neural /path/to/model.onnx
```

## How It Works

### The Distillation Pipeline
```
Entropic Source → Triplet DNA Gates → Neural Gate → File Copy → Negentropic Target
```

Each Python file in the source is analyzed through three layers:

### 1. Triplet DNA Gates
- **Inertia Gate**: Rejects disconnected "island" code with no dependencies
- **Entelecheia Gate**: Ensures code serves a clear purpose (LTL spec)
- **Homeostasis Gate**: Blocks transfers when system entropy is too high

### 2. Neural Gate
Analyzes code using 5 features:
- **Complexity**: McCabe cyclomatic complexity (decision points)
- **Depth**: Maximum AST nesting level
- **Similarity**: Semantic duplicate detection (future)
- **Churn**: File size relative to norms
- **Anti-Patterns**: Detection of "Manager", "AbstractFactory", God Classes, etc.

**Thrash Prediction**: Files with >80% thrash probability are rejected.

### 3. File Transfer
Accepted files are copied to the target directory, preserving structure.

## Use Cases

### 1. Legacy Code Cleanup
**Scenario**: You have a legacy repository with years of accumulated cruft.

```bash
# Distill only the clean, essential code
python3 -m nucleus.tools.distillation.distill_engine \
    --source ~/legacy_project \
    --target ~/legacy_project_clean
```

**Result**: Anti-patterns, dead code islands, and overly complex modules are filtered out.

### 2. Multi-Repository Merge
**Scenario**: Merging multiple repos with overlapping functionality.

```bash
# Merge repo A
python3 -m nucleus.tools.distillation.distill_engine \
    --source ~/repo_a \
    --target ~/merged_repo

# Merge repo B (duplicates auto-rejected)
python3 -m nucleus.tools.distillation.distill_engine \
    --source ~/repo_b \
    --target ~/merged_repo
```

**Result**: The Neural Gate's similarity detection prevents duplicate code.

### 3. AI-Generated Code Validation
**Scenario**: Filtering LLM-generated code before integration.

```bash
# Validate AI output
python3 -m nucleus.tools.distillation.distill_engine \
    --source ~/ai_generated_code \
    --target ~/validated_code
```

**Result**: Over-engineered or hallucinated code patterns are rejected.

### 4. Pre-Commit Hook Integration
**Scenario**: Automatic quality gating before commits.

```bash
# .git/hooks/pre-commit
#!/bin/bash
python3 -m nucleus.tools.distillation.distill_engine \
    --source . \
    --target /tmp/validated
```

**Result**: Only negentropic code reaches the repository.

## Emergency Stop

Create a `STOP_DISTILLATION` file in the target directory to immediately abort:

```bash
touch /path/to/negentropic/repo/STOP_DISTILLATION
```

## Output Interpretation

### Accepted File
```
Processing candidate: utils/helpers.py
Neural Analysis: complexity=0.15, depth=0.20, anti_pattern=0.00, thrash_prob=0.12
✅ ACCEPTED & COPIED: utils/helpers.py -> /target/utils/helpers.py
```

### Rejected File (Anti-Pattern)
```
Processing candidate: managers/user_manager.py
Neural Analysis: complexity=0.45, depth=0.30, anti_pattern=0.60, thrash_prob=0.85
❌ REJECTED by NeuralGate (Prob 0.85): managers/user_manager.py
```

### Rejected File (DNA Gate)
```
Processing candidate: isolated_util.py
❌ REJECTED by Inertia: isolated_util.py
```

## Advanced Configuration

### Custom Anti-Patterns
Edit `neural_adapter.py` to add project-specific patterns:

```python
self.anti_patterns = [
    "Manager",
    "Handler",  # Add custom pattern
    "YourCompanyAntiPattern"
]
```

### Adjust Complexity Thresholds
Modify thresholds in `neural_adapter.py`:

```python
if complexity > 0.7 and depth > 0.7:  # Adjust these
    return 0.95
```

## Testing

Run the test suite:

```bash
cd /Users/kroma/pluribus
python3 -m unittest nucleus.tools.distillation.test_distillation
```

## Troubleshooting

### All Files Rejected
- Check that source contains valid Python files
- Lower the thrash threshold in `distill_engine.py` (line 82)

### No Files Copied
- Verify target directory exists and is writable
- Check logs for specific rejection reasons

### Syntax Errors
- Distiller automatically rejects unparseable files
- Fix syntax in source before running

## Architecture

See detailed documentation:
- [Production Architecture](file:///Users/kroma/.gemini/antigravity/brain/8ac136c0-6c85-4ca6-b637-d9a36925cd17/production_distillation_architecture.md)
- [Neural Gate Design](file:///Users/kroma/.gemini/antigravity/brain/8ac136c0-6c85-4ca6-b637-d9a36925cd17/neural_gate_design.md)
