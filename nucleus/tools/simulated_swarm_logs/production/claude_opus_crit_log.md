
# Agent Log: Claude Opus (Critic)
## Mission: Neural Gate Design & Attack Surfaces
**Identity**: Ring 3 Critic | **Mode**: Production Design | **Session**: PBTSO-003

### 1. The Neural Gate Concept
We want to replace `if h_struct > 0.5` with `model.predict(patch)`.
Why? Because "Thrash" is subtle. It's not just "High Complexity". It's "Cyclical Churn". A heuristic can't see circles in time easily. A massive Transformer (or RNN) can.

**Feature Vector**:
- **AST Bag-of-Nodes**: Counts of `If`, `For`, `Class`.
- **Entropy Vector ($H^*$ of patch)**: The 8 dimensions we already track.
- **Context Vector**: Embedding of the `Target` file.
- **History Vector**: Embedding of the last 5 patches to this file (To detect loops).

**Model Architecture**:
- **Input**: Sequence of (Patch, Context).
- **Output**: $P(\text{Thrash})$.
- **Backbone**: Lightweight Transformer (DistilBERT or similar, fine-tuned on "Bad Agent Logs").

### 2. Training Strategy (The Data Problem)
We don't have labeled "Thrash" data.
**Solution**: Synthetic Adversarial Generation.
- **Generator (Red Team)**: An agent explicitly rewarded for generating "Valid but Useless" code (churn).
- **Discriminator (Blue Team)**: The Neural Gate.
- **Training Loop**: Run `distill_repo.py` in "Adversarial Mode" to train the gate.

### 3. Integration Risk
What if the Neural Gate hallucinates and blocks a critical fix?
**Mitigation**: The **Hybrid Gate**.
$$ \text{Gate} = (\text{Alpha} \times \text{Heuristic}) + (\text{Beta} \times \text{Neural}) $$
If Heuristic says "SAFE" and Neural says "UNSAFE", we flag for Human Review (Active Learning).
If Heuristic says "UNSAFE", we block immediately (Fast Path).

### 4. Recommendation
Implement `neural_adapter.py` as an optional plugin.
Start with `HeuristicGate` (v1) and log data for v2 Training.
