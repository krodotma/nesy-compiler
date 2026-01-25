
# core.py - Reactive Mutator Orchestrator
# Part of Reactive Evolution v1
# "The Cell Cycle"

import logging
from .ltl_monitor import ltl_monitor
from .grammar_filter import GrammarFilter
from .entropy_gate import EntropyGate, UncertaintyVector, EntropyVector
from .inertia_rank import InertiaRank
from .active_synthesizer import ActiveSynthesizer

logger = logging.getLogger("ReactiveMutator")

class ReactiveMutator:
    """
    The Main Evolutionary Engine.
    Executes the Cell Cycle: G1 -> S -> G2 -> M.
    """

    def __init__(self, root_dir: str, neural_adapter=None):
        self.grammar = GrammarFilter()
        self.gate = EntropyGate()
        self.inertia = InertiaRank(root_dir)
        self.synthesizer = ActiveSynthesizer()
        self.neural = neural_adapter

    @ltl_monitor("G(cycle_duration < 10.0)")
    def run_cycle(self, target_file: str, event_context: str, 
                  uncertainty: UncertaintyVector, entropy: EntropyVector):
        """
        Execute one Evolutionary Step.
        """
        
        # --- Phase G1: Observation ---
        if not self.gate.check_g1_availability(uncertainty):
            logger.warning("G1 Fail: Environment too unstable (High Aleatoric).")
            return "SLEEP"

        # --- Phase S: Synthesis ---
        ltl_spec = self.synthesizer.derive_spec(event_context)
        candidate = self.synthesizer.synthesize(target_file, ltl_spec, self.grammar)
        
        if not candidate:
            logger.info("S Fail: Could not synthesize valid candidate.")
            return "ABORT_S"

        # --- Phase G2: Verification ---
        rank = self.inertia.get_rank(target_file)
        if not self.gate.check_g2_safety(entropy, uncertainty, rank):
            logger.warning(f"G2 Fail: Safety Violation (Rank {rank:.2f}).")
            return "ABORT_G2"

        # --- Phase M: Mitosis ---
        self._apply_patch(candidate)
        return "SUCCESS"

    def _apply_patch(self, candidate):
        logger.info(f"Applying patch to {candidate.source_file}")
        # In real implementation: file write
        print(f"[MITOSIS] Applied Grid: {candidate.description}")
