
# Agent Log: Codex (Synthesizer)
## Mission: Distill Engine & CLI Design
**Identity**: Ring 3 Synthesizer | **Mode**: Production CodeGen | **Session**: PBTSO-003

### 1. `distill_engine.py` (The Walker)
This needs to be robust, parallel, and resumable.

```python
import click
from concurrent.futures import ThreadPoolExecutor
from .reactive_mutator.core import ReactiveMutator

@click.command()
@click.option("--source", required=True, help="Entropic Source path")
@click.option("--target", required=True, help="Negentropic Target path")
@click.option("--dna-level", default="strict", type=click.Choice(["relaxed", "strict", "neural"]))
def distill(source, target, dna_level):
    """
    Main Entry Point.
    Walks source, filters via Reactive Mutator, merges to Target.
    """
    engine = DistillationEngine(target, dna_level)
    report = engine.process(source)
    print(report.summary())
```

### 2. The `NeuralAdapter` Interface
We need a clean interface to plug in Torch/ONNX later.

```python
class NeuralAdapter:
    def __init__(self, model_path=None):
        self.model = self.load_model(model_path) if model_path else None
        
    def predict_thrash(self, feature_vector):
        if not self.model:
            return 0.5 # Neural Indifference (Fallback to Heuristic)
        
        tensor = self.vectorize(feature_vector)
        return self.model.forward(tensor)
```

### 3. Manifest Integration
The user wants this to be "Interfaceable".
We should register it in `pluribus/MANIFEST.yaml`.

```yaml
tools:
  distiller:
    path: nucleus/tools/distillation/distill_engine.py
    interface: cli
    capabilities: [code_cleaning, feature_migration]
    dna_support: [inertia, entelecheia, homeostasis]
```

### 4. Migration Plan
I will refactor `verify_cycle.py` logic into `distill_engine.py`.
I will make `ReactiveMutator` a dependency of `DistillationEngine`.
Distiller is the *Caller*. Mutator is the *Worker*.
