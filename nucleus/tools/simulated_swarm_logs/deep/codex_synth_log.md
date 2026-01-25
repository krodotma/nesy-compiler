
# Agent Log: Codex (Synthesizer)
## Mission: LTL Code Stubs & Reactive Mutator
**Identity**: Ring 3 Synthesizer | **Mode**: Deep CodeGen | **Session**: PBTSO-002

### 1. `ltl_spec.py`
Defining the data structures for LTL properties in Python.
We can't import a full LTL solver (like Spot/Strix) easily in Python, so we define the **Interfaces** and mock the solver via the LLM (Synthesizer).

```python
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class LTLProperty:
    formula: str  # e.g., "G(drift -> F(patch))"
    description: str
    
    def check(self, trace: List[Event]) -> bool:
        """
        Verify if a trace satisfies the property.
        (Mock implementation of logic checker)
        """
        pass

class SafetySpec(LTLProperty):
    """ Safety: Something bad never happens. """
    pass

class LivenessSpec(LTLProperty):
    """ Liveness: Something good eventually happens. """
    pass
```

### 2. `reactive_mutator.py`
The Loop that implements the Santolucito Cycle.

```python
class ReactiveMutator:
    def __init__(self, grammar: GrammarFilter, inertia: InertiaRank):
        self.grammar = grammar
        self.inertia = inertia
        self.specs: List[LTLProperty] = []

    def observe(self, system_state: EntropyVector):
        """ Read the H* and U* vectors """
        if system_state.h_alea > 0.5:
            # Aleatoric Gap: Do nothing (Robustness)
            return None

        if system_state.h_epis > 0.5:
            # Epistemic Gap: Synthesize a Probe/Test
            return self.synthesize_probe()

        # Normal Operation: Synthesize Mutation
        return self.synthesize_mutation()

    def synthesize_mutation(self):
        """
        1. Generate Candidate (via Grammar)
        2. Verify against LTL Specs (Model Checking)
        3. Check Inertia Safety
        """
        candidate = self.grammar.generate()
        
        # Verify LTL
        if not self.verify_candidate(candidate):
            return None # Reject (Anti-Thrash)
            
        return candidate
```

### 3. Bridging to `laser/uncertainty.py`
```python
from laser.uncertainty import ConfidenceScorer

def get_risk_profile(target_file):
    scorer = ConfidenceScorer()
    # Mocking a "read" of the file's stability
    # In reality, this links to the OHM history
    latency_variance = ohm.get_latency_variance(target_file)
    model_disagreement = ohm.get_model_disagreement(target_file)
    
    # Map to H_alea and H_epis
    # ...
```
